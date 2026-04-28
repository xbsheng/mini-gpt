import tiktoken
import torch
from numpy import arange
from torch import nn

from config import DATA_DIR
from dataloader import create_dataloader
from embedding import EMBEDDING_DIM, token_embedding_layer

# 全局设置：禁用科学计数法
torch.set_printoptions(sci_mode=False)


class CausalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias=False, dropout=0.2):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.mT

        # 因果掩码
        mask = torch.triu(
            torch.ones(attn_scores.shape[-2:]),
            diagonal=1,
        ).to(x.device, dtype=torch.bool)
        print("mask", mask)
        # tensor([[0., 1., 1.],
        #         [0., 0., 1.],
        #         [0., 0., 0.]])

        masked_attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        print("masked_attn_scores", masked_attn_scores)
        # tensor([[ -0.2558,     -inf,     -inf],
        #         [  3.2698, -10.0323,     -inf],
        #         [  2.9594,  -4.3804,  -8.4760]], grad_fn=<MaskedFillBackward0>)

        d_k = keys.shape[-1]
        # 缩放点积注意力(scaled dot-product attention)
        attn_weights = torch.softmax(masked_attn_scores / (d_k**0.5), dim=-1)
        print("attn_weights", attn_weights)
        # tensor([[1.0000, 0.0000, 0.0000],
        #         [0.6966, 0.3034, 0.0000],
        #         [0.4714, 0.2980, 0.2307]], grad_fn=<SoftmaxBackward0>)

        attn_weights = self.dropout(attn_weights)
        print("dropout attn_weights:", attn_weights)
        # tensor([[1.2500, 0.0000, 0.0000],
        #         [0.8708, 0.0000, 0.0000],
        #         [0.0000, 0.3724, 0.2883]], grad_fn=<MulBackward0>)

        context_vec = attn_weights @ values

        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_in: int,
        d_out: int,
        qkv_bias=False,
        dropout=0.2,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in=d_in,
                    d_out=d_out,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        return torch.cat([head(x) for head in self.heads], dim=-1)


def handle_single_text():
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = tokenizer.encode("how are you")  # [4919, 389, 345]
    token_embeddings = token_embedding_layer(torch.tensor(inputs))
    print(token_embeddings.shape)  # torch.Size([3, 256])

    attention = CausalAttention(EMBEDDING_DIM, EMBEDDING_DIM)
    context_vec = attention(token_embeddings)  # torch.Size([3, 256])
    print(context_vec.shape)


def handle_batch():
    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        BATCH_SIZE = 2
        MAX_CONTENT = 4

        pos_embedding_layer = torch.nn.Embedding(
            num_embeddings=MAX_CONTENT,
            embedding_dim=EMBEDDING_DIM,
        )

        dataloader = create_dataloader(
            raw_text,
            batch_size=BATCH_SIZE,
            max_content=MAX_CONTENT,
            stride=1,
            shuffle=False,
        )

        for i, (inputs, targets) in enumerate(dataloader):
            if i > 0:
                break

            # inputs: torch.Size([2, 4])

            token_embeddings = token_embedding_layer(inputs)  # torch.Size([2, 4, 256])
            pos_embeddings = pos_embedding_layer(torch.arange(MAX_CONTENT))

            embeddings = token_embeddings + pos_embeddings
            print(embeddings.shape)

            attention = CausalAttention(EMBEDDING_DIM, EMBEDDING_DIM)
            context_vec = attention(embeddings)
            print("single head:", context_vec.shape)  # torch.Size([2, 4, 256])

            multi_head_attention = MultiHeadAttentionWrapper(
                3, EMBEDDING_DIM, EMBEDDING_DIM
            )
            multi_head_context_vec = multi_head_attention(embeddings)
            print("multi_head_context_vec:", multi_head_context_vec.shape)
            # torch.Size([2, 4, 768])


if __name__ == "__main__":
    # handle_single_text()

    handle_batch()
