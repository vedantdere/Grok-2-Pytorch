import torch
from torch import nn
from torch.nn import functional as F
from attn_implementation import eager_paged_attention_forward

class GrokRMSNorm(nn.Module):
    def __init__(self,
                 dim,
                 eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        variance = x.pow(2).mean(-1,keepdim=True)
        x = x * torch.sqrt(variance + self.eps)
        x = self.weight * x
        return x

class Grok1MLP(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_id,
                 ):
        super().__init__()

        self.gate_up_proj = nn.Linear(
            hidden_size,
            sum([intermediate_size] * 2),
            bias=False
        )

        self.down_proj = nn.Linear(
            sum([intermediate_size] * 2),
            hidden_size,
            bias=False
        )

        self.act_fn = nn.GELU(approximate="tanh")
        self.layer_id = layer_id

    def forward(self,x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x
    
class GrokExperts(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))

        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

        self.alpha = 1.702
        self.limit = 7.0

    def forward(self,
                hidden_state,
                router_indices,
                routing_weights):
        """"
        Apply the experts to the hidden state based on the routing indices and weights.
        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            router_indices (torch.Tensor): Indices of the experts to route to, shape (batch_size, num_experts).
            routing_weights (torch.Tensor): Weights for the routing, shape (batch_size, num_experts).
        Returns:
            torch.Tensor: Output tensor after applying the experts, shape (batch_size, sequence_length, hidden_size).
        """
        
        batch_size = hidden_state.shape[0]
        hidden_state = hidden_state.view(-1,self.hidden_size)
        num_experts = routing_weights.shape[1]

        if self.training:
            next_states = torch.zeros_like(hidden_state,dtype=hidden_state.dtype,device=hidden_state.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices,num_classes=num_experts)
                expert_mask = expert_mask.permute(2,1,0)

                expert_hit = torch.greater(expert_mask.sum(dim=(-1,-2)),0).nonzero()
            
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_state[token_idx]

                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate,up = gate_up[...,::2],gate_up[...,1:2]
                gate = gate.clamp(min=None,max=self.limit)

                up = up.clamp(min=-self.limit,max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)

                gated_output = (up + 1) * glu

                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx,expert_idx,None]
                next_states.index_add_(0,token_idx,weighted_output.to(hidden_state.dtype))
            next_states = next_states.view(batch_size,-1,self.hidden_size)

        
        else:
            hidden_state = hidden_state.repeat(num_experts,1)
            hidden_state = hidden_state.view(num_experts,-1,self.hidden_size)

            gate_up = torch.bmm(hidden_state,self.gate_up_proj) + self.gate_up_proj_bias[...,None,:]
            gate,up = gate_up[...,::2],gate_up[...,1::2]
            gate = gate.clamp(min=None,max=self.limit)
            up = up.clamp(min=-self.limit,max=self.limit)

            glu = gate * torch.sigmoid(gate * self.alpha)

            next_states = torch.bmm(((up + 1)*glu),self.down_proj)

            next_states = next_states + self.down_proj_bias[...,None,:]
            next_states = next_states.view(num_experts,batch_size,-1,self.hidden_size)
            next_states = next_states * routing_weights.transpose(0,1).view(num_experts,batch_size,-1)[...,None]
            next_states = next_states.sum(dim=0)
        return next_states
    
class GrokTopKRouter(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts

        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts,self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self,hidden_state):
        """"
        Apply the top-k routing to the hidden state.
        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
        Returns:
            torch.Tensor: Router scores of shape (batch_size, sequence_length, num_experts).
            torch.Tensor: Indices of the top-k experts for each token, shape (batch_size, sequence_length, top_k).
        """
        hidden_state = hidden_state.reshape(-1,self.hidden_dim)
        router_logits = F.linear(hidden_state,self.weight,self.bias)

        router_top_value,router_indices = torch.topk(router_logits,self.top_k,dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value,dim=-1,dtype=router_top_value.dtype)

        router_scores = torch.zeros_like(router_logits).scatter_(1,router_indices,router_top_value)

        return router_scores,router_indices
    
class GrokMLP(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.router = GrokTopKRouter(config)
        self.experts = GrokExperts(config)

    def forward(self,hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores
    
class Grok1MoE(nn.Module):
    def __init__(self,
                 config,
                 layer_id,
                 num_experts,
                 top_k,
                 hidden_size,
                 intermediate_size,
                 ):
        super().__init__()

        self.hidden_size = hidden_size

        self.gate = nn.Linear(
            self.hidden_size,
            num_experts,
            bias=False
        )

        self.router_logit_softcapping = config.router_logit_softcapping

        self.experts = GrokMLP(config)
    
    def forward(self,
                hidden_states):
        topk_output = self.experts(hidden_states)
        return topk_output
        

class GptOssRotartEmbedding(nn.Module):
    def __init__(self,
                config,
                device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq,self.attenstion_scaling = self.rope_init_fn(self.config,device)

        self.register_buffer("inv_freq",inv_freq)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self,x,position_ids):
        """"
        Apply Rotary Position Embeddings to the input tensor x based on position_ids.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            position_ids (torch.Tensor): Position IDs of shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: Cosine and sine embeddings for the rotary position embeddings.
        """
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,1).to(x.device)
        postion_ids_expanded = position_ids[:,None,:].float()

        device_type = x.device.type if isinstance(x.device.type,str) and x.device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type,enabled=False):
            freqs = (inv_freq_expanded.float() @ postion_ids_expanded.float()).transpose(1,2)
            emb = freqs
            cos = emb.cos() * self.attenstion_scaling
            sin = emb.sin() * self.attenstion_scaling

        return cos.to(x.dtype),sin.to(x.dtype)


def _apply_rotary_emb(x,
                    cos,
                    sin):
    first_half,second_half = torch.chunk(x,2,dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_,second_),dim=-1)


def apply_rotart_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = _apply_rotary_emb(q,cos,sin)
    k_embed = _apply_rotary_emb(k,cos,sin)
    return q_embed,k_embed


class Grok1Attention(nn.Module):
    def __init__(self,
                 config,
                 hidden_size,
                 num_heads,
                 num_kv_heads,
                 layer_id,
                 max_position,
                 rope_theta
                 ):
        super().__init__()

        self.config = config
        self.layer_id = layer_id
        
        self.head_dim = getattr(config, "head_dim", config.hidden_size  // config.num_attention_heads)

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim *-0.5

        self.attention_dropout = config.attention_dropout

        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.bias
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.bias
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.bias
        )        

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias
        )

        self.sliding_window = config.sliding_window if config.layer_types[layer_id]=="sliding_attention" else None
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    def forward(self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                **kwargs):
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1,2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotart_pos_emb(query_states, key_states, cos, sin)

        attention_inference = eager_paged_attention_forward

        attn_output, attn_weight = attention_inference(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks
        )

        attn_output = attn_output.reshape(*input_shape,-1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weight
    

class Grok1DecoderLayer(nn.Module):
    def __init__(self,
                 config,
                 layer_id
                 ):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.residual_moe = getattr(config, "residual_moe", False)
        self.layer_id = layer_id

        # self.alt_stream = torch.cuda.Stream()

        rope_theta = getattr(config, "rope_theta", 10000)

        self.self_attn = Grok1Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.context_len,
            num_kv_heads = config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
        )

        if self.num_experts > 0:
            self.block_sparse_moe = Grok1MoE(
                config=config,
                layer_id=layer_id,
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )

            if self.residual_moe:
                self.mlp = Grok1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    layer_id=layer_id
                )

        self.pre_attn_norm = GrokRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attn_norm = GrokRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.pre_moe_norm = GrokRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_moe_norm = GrokRMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        if self.num_experts > 0:
            if self.residual_moe:
                self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.block_sparse_moe

    def forward(self,
                positions,
                hidden_states,
                residual=None,
                deferred_norm=None):
        
        hidden_states_original = hidden_states
        residual_original = residual

        hidden_states = self.pre_attn_norm(hidden_states)
        residual = self.pre_attn_norm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = self.post_attn_norm(hidden_states)
        residual = self.post_attn_norm(residual)

        return hidden_states, residual, self.post_moe_norm


    def moe_with_rmoe(self,x):
        # current_stream = torch.cuda.current_stream()
        # self.atl_stream.wait_stream(current_stream)

        mlp_result = self.mlp(x)

        # with torch.cuda.stream(self.alt_stream):
        moe_result = self.block_sparse_moe(x)
        
        return (mlp_result + moe_result) / 1.4142135623730951
    

class Grok1Model(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_idx
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [
                Grok1DecoderLayer(
                    config,
                    i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size,self.padding_idx)

        self.norm = GrokRMSNorm(config.hidden_size,
                                eps=config.rms_norm_eps)
        
    def forward(self,
                input_ids,
                positions,
                input_embeds=None):
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        residual, deferred_norm = None, None

        for i in range(len(self.layers)):
            hidden_states, residual, deferred_norm = self.layers[i](
                positions,
                hidden_states,
                residual,
                deferred_norm
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Grok1ModelForCausalLM(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.config = config
        
        self.model = Grok1Model(
            config=config
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self,
                input_ids,
                positions=None,
                input_embeds=None):
        
        hidden_states = self.model(input_ids, positions, input_embeds)
        out = self.lm_head(hidden_states)
        return out