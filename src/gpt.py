from dataclasses import dataclass

import torch
from torch import nn

from config import DATA_DIR
from dataloader import create_dataloader
from utils import set_seed

set_seed()
torch.set_printoptions(sci_mode=False)  # 关闭科学计数法


@dataclass
class GPT_CONFIG_124M:
    vocab_size = 50257  # 词汇表大小
    context_length = 1024  # 上下文长度
    emb_dim = 768  # 嵌入维度
    n_heads = 12  # 注意力头的数量
    n_layers = 12  # 层数
    drop_rate = 0.1  # dropout率
    qkv_bias = False  # 查询-键-值偏置


class DummyGPTModel(nn.Module):
    def __init__(self, config: GPT_CONFIG_124M):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_embedding = nn.Embedding(config.context_length, config.emb_dim)

        self.dropout = nn.Dropout(config.drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock() for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNorm(config.emb_dim)

        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape

        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=input_ids.device))
        x = token_embeds + pos_embeds

        x = self.dropout(x)
        x = self.transformer_blocks(x)

        x = self.final_norm(x)

        return self.out_head(x)


class DummyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


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


if __name__ == "__main__":
    config = GPT_CONFIG_124M()
    model = DummyGPTModel(config)

    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        dataloader = create_dataloader(
            raw_text,
            batch_size=4,
            max_content=config.context_length,
            stride=1,
            shuffle=False,
        )

        for i, (inputs, targets) in enumerate(dataloader):
            if i > 0:
                break

            print(inputs.shape, (inputs, targets))
            # inputs.shape: torch.Size([4, 1024])

            outputs = model(inputs)
            print("outputs.shape", outputs.shape)  # (batch_size, seq_len, vocab_size)
            # torch.Size([4, 1024, 50257])
            # torch.Size([4, 1024, 50257])
