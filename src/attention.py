import tiktoken
import torch
from torch import nn

from embedding import EMBEDDING_DIM, token_embedding_layer

# 全局设置：禁用科学计数法
torch.set_printoptions(sci_mode=False)


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        # 缩放点积注意力(scaled dot-product attention)
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        context_vec = attn_weights @ values

        return context_vec


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = tokenizer.encode("how are you")  # [4919, 389, 345]
    token_embeddings = token_embedding_layer(torch.tensor(inputs))
    print(token_embeddings.shape)  # torch.Size([3, 256])

    attention = SelfAttention(EMBEDDING_DIM, EMBEDDING_DIM)
    context_vec = attention(token_embeddings)  # torch.Size([3, 256])
    print(context_vec.shape)
