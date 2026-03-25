import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

def rms(x: Tensor, eps: float):
    return x*torch.sqrt(x.pow(2).mean(dim=1, keepdim=True)+eps)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    
    def forward(self, x:Tensor)->Tensor:
        return rms(x, self.eps)*self.weight


# transformer

class PreNorm(nn.Module):
    def __init__(self, dim:int, fn:nn.Module, eps:float):
        self.norm=RMSNorm(dim, eps=eps)
        self.fn=fn

    def forward(self, x:Tensor)->Tensor:
        return self.fn(self.norm(x))

class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads:int=8, dim_head:int=64, dropout: float=0.0):
        inner_dim=heads*dim_head
        self.heads=heads
        self.dim_head=dim_head
        self.dropout=dropout

        self.to_qkv=nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out=nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: Tensor)->Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        def split_heads(y: Tensor)->Tensor:
            return rearrange(y, "b t (h d) -> b h t d", h=self.heads)
        
        q, k, v = map(split_heads, (q, k, v))

        out=F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        out=rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)