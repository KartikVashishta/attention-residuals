import torch
from torch import Tensor, nn
import torch.nn.attention as F

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
        return self.fn(self.fn(x))