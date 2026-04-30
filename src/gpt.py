from dataclasses import dataclass

import torch
from torch import nn

from config import DATA_DIR
from dataloader import create_dataloader
from gelu import GELU
from normal import LayerNorm
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


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()

        hidden_dim = 4 * emb_dim

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


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
