import torch
from torch import nn
from layers.linear_layer import ReplicatedLinear, ColumnParallelLinear, MergedColumnParallelLinear,RowParallelLinear


class Grok1MLP(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_id,
                 quant_config=None,
                 prefix="",
                 reduce_results=True,
                 use_presharded_weights=False,
                 split_gate_up=False):
        super().__init__()
        
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_predix("gate_up_proj", prefix),
            use_presharded_weights=use_presharded_weights
        )

        
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            use_presharded_weights=use_presharded_weights
        )

        self.act_fn = nn.ReLU()
        self.layer_id = layer_id

    def forward(self,x):
        gate_up, _ = self.gate_up_proj(x)
        x , _ = self.act_fn(x)
        x , _ = self.down_proj(x)
        return x

class Experts(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts,self.hidden_size,2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts,2 * self.expert_dim))

        self.down_proj = nn.Parameter(torch.empty((self.num_experts,self.expert_dim,self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts,self.hidden_size))

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



class Grok1MoE(nn.Module):
    def __init__(self,
                 config,
                 layer_id,
                 num_experts,
                 top_k,
                 hidden_sizes,
                 intermediate_size,
                 params_dtype=None,
                 quant_config=None,
                 tp_size=None,
                 use_presharded_weights=False,
                 inplace=True,
                 no_combine=False,
                 prefix=""):
        super.__init__()

        self.gate = ReplicatedLinear(
            hidden_sizes,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None
        )

        self.router_logit_softcapping = getattr(
            config, "router_logit_softcapping", 30.0
        )

        custom_routing_function = function.partial(
            fused_moe_router_shim, self.router_logit_softcapping
        )

        