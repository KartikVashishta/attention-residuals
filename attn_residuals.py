import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

def rms(x: Tensor, eps: float):
    return x*torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+eps)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    
    def forward(self, x:Tensor)->Tensor:
        return rms(x, self.eps)*self.weight

class DepthResidual(nn.Module):
    """
        h_l = sum_i softmax_i(w_l^T RMSNorm(v_i))*v_i

        Keep query and RMSNorm gain separate
        Since q^T (gamma * RMS(v)) == (q * gamma)^T RMS(v),
        we can fold gamma into q for scoring.
    """
    def __init__(self, dim:int, eps: float=1e-8, zero_init:bool=True):
        super().__init__()
        self.query=nn.Parameter(torch.zeros(dim))
        self.norm=RMSNorm(dim, eps=eps)

        if not zero_init:
            nn.init.normal_(self.query, std=0.02)

    def logits(self, sources: Tensor |list[Tensor]| tuple[Tensor,...])->Tensor:
        sources=stack_layers(sources) # [n, b, t, d]
        q=(self.query*self.norm.weight).float() #[d]
        k = rms(sources.float, self.norm.eps) # [n, b, t, d]
        return torch.einsum("d, n b t d -> n b t", q, k)

    def forward(self, sources: Tensor | list[Tensor] | tuple[Tensor, ...])->Tensor:
        sources=stack_layers(sources)
        weights=self.logits(sources).softmax(dim=0)
        out=torch.einsum('n b t, n b t d ->', weights, sources.float())
        return out.to(sources.dtype)

# transformer
class PreNorm(nn.Module):
    def __init__(self, dim:int, fn:nn.Module, eps:float):
        super().__init__()
        self.norm=RMSNorm(dim, eps=eps)
        self.fn=fn

    def forward(self, x:Tensor)->Tensor:
        return self.fn(self.norm(x))

class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads:int=8, dim_head:int=64, dropout: float=0.0):
        super().__init__()
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

class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult:int=4, dropout:float=0.0):
        # dropout not needed unless training on a smaller training data
        super().__init__()
        inner_dim=dim*mult
        self.to_hidden=nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out=nn.Linear(inner_dim,dim,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x:Tensor)->Tensor:
        gate, value = self.to_hidden(x).chunk(2,dim=-1)
        x = F.silu(gate)*value
        x = self.dropout(x)
        return self.to_out(x)

# helpers

def stack_layers(sources: Tensor |
                list[Tensor] |
                tuple[Tensor, ...])->Tensor:
    if isinstance(sources, Tensor):
        assert sources.ndim==4, f'expected [n, b, t, d] got {tuple(sources.shape)}'
        return sources
    assert len(sources)>0, 'needs at least one source'
    return torch.stack(tuple(sources), dim=0)

