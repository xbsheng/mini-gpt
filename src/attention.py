import torch
from torch import nn

from config import DATA_DIR
from dataloader import create_dataloader
from embedding import EMBEDDING_DIM, token_embedding_layer

# 全局设置：禁用科学计数法
torch.set_printoptions(sci_mode=False)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_in: int,
        d_out: int,
        qkv_bias=False,
        dropout=0.2,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out % num_heads != 0"

        self.num_heads = num_heads
        self.d_head = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_embed = x.shape  # 2 4 256
        num_heads = self.num_heads
        d_head = self.d_head

        reshape_shape = (batch_size, seq_len, num_heads, d_head)
        queries = self.W_query(x).reshape(reshape_shape).transpose(1, 2)
        # batch_size seq_len d_out
        # reshape      -> batch_size seq_len num_heads d_head
        # transpose -> batch_size num_heads seq_len d_head
        keys = self.W_key(x).reshape(reshape_shape).transpose(1, 2)
        values = self.W_value(x).reshape(reshape_shape).transpose(1, 2)

        print("queries.shape:", queries.shape)  # torch.Size([2, 8, 4, 32])

        attn_scores = queries @ keys.mT
        print("attn_scores", attn_scores.shape)
        # [batch_size, num_heads, seq_len, seq_len]: torch.Size([2, 8, 4, 4])

        # 因果掩码
        mask = torch.triu(
            torch.ones(attn_scores.shape[-2:]),
            diagonal=1,
        ).to(x.device, dtype=torch.bool)

        masked_attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        d_k = keys.shape[-1]
        # 缩放点积注意力(scaled dot-product attention)
        attn_weights = torch.softmax(masked_attn_scores / (d_k**0.5), dim=-1)
        # [batch_size, num_heads, seq_len, seq_len]

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, d_head]
        #                 -> [batch_size, num_heads, seq_len, d_head]: [2, 8, 4, 32]
        # transpose(1, 2) -> [batch_size, seq_len, num_heads, d_head]: [2, 4, 8, 32]
        print("context_vec.shape", context_vec.shape)
        # torch.Size([2, 4, 8, 32])

        context_vec = context_vec.flatten(start_dim=-2)
        print("context_vec.shape2", context_vec.shape)
        # torch.Size([2, 4, 256])

        context_vec = self.out_proj(context_vec)

        return context_vec


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
            print("embeddings.shape:", embeddings.shape)
            # torch.Size([2, 4, 256])

            multi_head_attention = MultiHeadAttention(8, EMBEDDING_DIM, EMBEDDING_DIM)
            multi_head_context_vec = multi_head_attention(embeddings)
            print("multi_head_context_vec:", multi_head_context_vec.shape)
            # torch.Size([2, 4, 256])


if __name__ == "__main__":
    handle_batch()
