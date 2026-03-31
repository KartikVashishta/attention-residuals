import torch
from attn_residuals import AttnResTransformer, BlockAttnResStack, CausalAttention, DepthResidual, FullAttnResStack, PreNorm, SwiGLU

def diff(a, b): return (a-b).abs().max().item()

if __name__ == '__main__':
    
    torch.manual_seed(0)
    
    b, t, d, num_layers = 2, 5, 32, 4
    x = torch.randn(b, t, d)
    atomic_layers=[PreNorm(d, torch.nn.Linear(d, d, bias=False)) for _ in range(num_layers)]

    # full attnres: naive and the two phase schedule should match
    full = FullAttnResStack(d, atomic_layers)
    y_naive = full.forward_naive(x)
    y_two_phase = full.forward_two_phase(x, schedule_block_size=2)
    print('full max diff:', diff(y_naive,y_two_phase))

    # LM shape check
    model=AttnResTransformer(num_tokens=1024, dim=64, depth=4, max_seq_len=32, heads=4, dim_head=16,
    ff_mult=4, attnres='block', block_size=4)
    ids=torch.randint(0,1024,(2,16))
    logits=model(ids)
    print('lm logits shape', tuple(logits.shape))
    

