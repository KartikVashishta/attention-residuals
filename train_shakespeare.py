import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from attn_residuals import (AttnResTransformer,PreNorm,CausalAttention,SwiGLU,RMSNorm)


batch_size = 32
max_seq_len = 256
max_iters = 2500
eval_interval = 250
learning_rate = 1e-3
eval_iters = 100

# approx 15M params

dim = 384
depth = 6
heads = 6
dim_head = 64
ff_mult = 4
block_size = 4  #blockattn block size

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

#data prep

data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_path = 'input.txt'

if not os.path.exists(data_path):
    print("Downloading Tiny Shakespeare dataset..")
    with open(data_path, 'w', encoding='utf-8') as f: f.write(requests.get(data_url).text)

with open(data_path, 'r', encoding='utf-8') as f: text = f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)
print(f"Dataset characters: {len(text)}, Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode=lambda s: [stoi[c] for c in s]
decode=lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - max_seq_len, (batch_size,))
    x = torch.stack([data_split[i:i+max_seq_len] for i in ix])
    y = torch.stack([data_split[i+1:i+max_seq_len+1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            # reshape for cross entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = F.cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# baseline transformer

class StandardTransformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads, dim_head, ff_mult, eps=1e-8):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.layers = nn.ModuleList()
        for _ in range(depth): self.layers.append(nn.ModuleList([PreNorm(dim, CausalAttention(dim, heads, dim_head), eps), PreNorm(dim, SwiGLU(dim, ff_mult), eps)]))
            
        self.final_norm = RMSNorm(dim, eps)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, ids):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        
        for attn, ffn in self.layers:
            x = x+attn(x)
            x = x+ffn(x)
            
        x = self.final_norm(x)
        return self.to_logits(x)


###models
baseline_model = StandardTransformer(num_tokens=vocab_size, dim=dim, depth=depth, max_seq_len=max_seq_len,heads=heads, dim_head=dim_head, ff_mult=ff_mult).to(device)
attnres_model = AttnResTransformer(num_tokens=vocab_size, dim=dim, depth=depth, max_seq_len=max_seq_len,heads=heads, dim_head=dim_head, ff_mult=ff_mult, attnres='block', block_size=block_size).to(device)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Baseline parameters: {count_parameters(baseline_model):,}")
print(f"BlockAttnRes parameters: {count_parameters(attnres_model):,}")

# training loop
def train_model(model, name):
    print(f"\n## Training {name} ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    val_loss_history = []
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            val_loss_history.append((iter, losses['val']))
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        
        # determine if we should pass schedule_block_size for two-phase inference
        if isinstance(model, AttnResTransformer) and model.attnres == 'full':
             logits = model(xb, schedule_block_size=block_size)
        else:
             logits = model(xb)
             
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    return val_loss_history


baseline_history = train_model(baseline_model, "Standard Baseline")
attnres_history = train_model(attnres_model, "Block AttnRes (Your Impl)")

print("\nGenerating comparative plot...")
baseline_iters, baseline_losses = zip(*baseline_history)
attnres_iters, attnres_losses = zip(*attnres_history)

plt.figure(figsize=(10, 6))
plt.plot(baseline_iters, baseline_losses, label='Standard Residuals', marker='o')
plt.plot(attnres_iters, attnres_losses, label='Block AttnRes', marker='s')
plt.title('Validation Loss on Tiny Shakespeare (~15M Params)')
plt.xlabel('Training Iterations')
plt.ylabel('Cross Entropy Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plot_filename = 'loss_curve.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')