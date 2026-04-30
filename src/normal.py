import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 有偏方差

        print(
            "before norm:", "mean:", mean[0, 0, 0].item(), "var:", var[0, 0, 0].item()
        )
        # before norm: mean: 0.05480312183499336 var: 2.374007225036621

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        norm_x = self.scale * norm_x + self.shift

        print(
            "after norm:",
            "mean:",
            norm_x.mean(dim=-1, keepdim=True)[0, 0, 0].item(),
            "var:",
            norm_x.var(dim=-1, keepdim=True, unbiased=False)[0, 0, 0].item(),
        )
        # after norm: mean: -9.934107758624577e-09 var: 0.9999958276748657
        return norm_x
