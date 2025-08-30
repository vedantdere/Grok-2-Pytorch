import torch
from torch import nn
from torch.nn import functional as F

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

        self.mlp = Grok1MLP(config)
        

