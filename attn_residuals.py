import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

def exists(x): return x is not None
def rms(x: Tensor, eps: float): return x*torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+eps)

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

    def effective_query(self) -> Tensor:
        return (self.query*self.norm.weight).float()

    def logits(self, sources: Tensor |list[Tensor]| tuple[Tensor,...])->Tensor:
        sources=stack_layers(sources) # [n, b, t, d]
        q=self.effective_query() #[d]
        k = rms(sources.float(), self.norm.eps) # [n, b, t, d]
        return torch.einsum("d, n b t d -> n b t", q, k)

    def forward(self, sources: Tensor | list[Tensor] | tuple[Tensor, ...])->Tensor:
        sources=stack_layers(sources)
        weights=self.logits(sources).softmax(dim=0)
        out=torch.einsum('n b t, n b t d -> b t d', weights, sources.float())
        return out.to(sources.dtype)

class DepthResidualList(nn.Module):
    def __init__(self, dim: int, depth: int, eps: float, zero_init: bool = True):
        super().__init__()
        # for L layers (depth), create depth residual modules
        self.layers = nn.ModuleList([DepthResidual(dim, eps=eps, zero_init=zero_init) for _ in range(depth)])

    def __getitem__(self, idx:int) -> DepthResidual: return self.layers[idx]
    def __iter__(self, idx:int): return iter(self.layers)
    def __len__(self, idx:int): return len(self.layers)

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

# attnres stacks

class FullAttnResStack(nn.Module):
    def __init__(self, dim: int, layers, *, eps: float=1e-8, zero_init_queries:bool=True, is_final_aggregate:bool=True):
        super().__init__()
        self.layers=nn.ModuleList(list(layers))
        self.eps=eps
        
        depth=len(self.layers)
        self.residuals = DepthResidualList(dim, depth, eps, zero_init_queries)
        self.final_residual = DepthResidual(dim, eps, zero_init_queries) if is_final_aggregate else None
    
    def forward_naive(self, x: Tensor)->Tensor:
        sources = [x]
        for layer, residual in zip(self.layers, self.residuals):
            h = residual(sources)
            out = layer(h)
            sources.append(out)
        
        return self.final_residual(sources) if exists(self.final_residual) else sources[-1]

    def forward_two_phase(self, x: Tensor, schedule_block_size:int)->Tensor:
        assert schedule_block_size>0
        sources=[x]
        depth=len(self.layers)

        start=0
        while start<depth:
            end = min(start+schedule_block_size, depth)
            queries =torch.stack([self.residuals[i].effective_query() for i in range(start, end)], dim=0)
            inter_sources=stack_layers(sources)
            inter_stats = attn_with_stats(queries, inter_sources, self.eps)

            local_outputs=[] # outputs of intra-block
            for local_idx, layer_idx in enumerate(range(start, end)):
                stats = inter_stats.select(local_idx)
                if len(local_outputs)>0:
                    intra_sources = stack_layers(local_outputs)
                    intra = attn_with_stats(queries[local_idx:local_idx+1], intra_sources, self.eps).select(0)
                    stats = merge_attn_stats(stats, intra)
                h=stats.normalized()
                out = self.layers[layer_idx](h)
                local_outputs.append(out)
                sources.append(out)
            
            start=end
        
        return self.final_residual(sources) if exists(self.final_residual) else sources[-1]

    def forward(self, x: Tensor, schedule_block_size: int | None = None) -> Tensor:
        if schedule_block_size is None:
            return self.forward_naive(x)
        return self.forward_two_phase(x, schedule_block_size)

class BlockAttnResStack(nn.Module):
    pass

# helpers

def stack_layers(sources: Tensor |
                list[Tensor] |
                tuple[Tensor, ...])->Tensor:
    if isinstance(sources, Tensor):
        assert sources.ndim==4, f'expected [n, b, t, d] got {tuple(sources.shape)}'
        return sources
    assert len(sources)>0, 'needs at least one source'
    return torch.stack(tuple(sources), dim=0)

class SingleAttnStats:
    def __init__(self, numer: Tensor, denom: Tensor, max:Tensor):
        self.numer=numer #[b,t,d]
        self.max=max #[b,t]
        self.denom=denom #[b,t]

    def normalized(self) -> Tensor:
        return self.numer/self.denom[..., None]        

class AttnStats:
    # store the numerator => e^{s_{j}-m} * v_j where m is the max score so far
    # store the max m = max(s_j)
    # store the denominator sum_j e^{s_{j}-m}
    def __init__(self, numer: Tensor, denom: Tensor, max: Tensor):
        self.numer=numer #[q,b,t,d]
        self.max=max #[q,b,t]
        self.denom=denom #[q,b,t]
    
    def select(self, idx:int) -> 'SingleAttnStats':
        return SingleAttnStats(self.numer[idx], self.denom[idx], self.max[idx])

def attn_with_stats(queries: Tensor, sources: Tensor, eps: float=1e-8) -> AttnStats:
    """
    queries: [q, d]
    sources: [n, b, t, d]

    Returns the following for online softmax:
        numer = sum_i exp(logit_i - m)*v_i
        m = max_i logit_i
        denom = sum_i exp(logit_i - m)
    """
    normed = rms(sources, eps)
    logits = torch.einsum('q d, n b t d -> q n b t', queries,normed)
    m = logits.amax(dim=1)
    weights = torch.exp(logits - m[:, None])
    numer = torch.einsum('q n b t, n b t d -> q b t d', weights, sources)
    denom = weights.sum(dim=1)
    return AttnStats(numer, denom, m)

def single_source_stats(query: Tensor, source: Tensor, eps:float=1e-8) -> SingleAttnStats:
    score = torch.einsum('d, b t d -> b t', query, rms(source, eps))
    denom = torch.ones_like(score)
    return SingleAttnStats(source, score, denom)

def merge_attn_stats(a: SingleAttnStats, b: SingleAttnStats) -> SingleAttnStats:
    m=torch.maximum(a.max, b.max)
    wa=torch.exp(a.max - m)
    wb=torch.exp(b.max - m)
    numer=wa[..., None] * a.numer + wb[..., None] * b.numer
    denom=wa * a.denom + wb * b.denom
    return SingleAttnStats(numer, m, denom)
