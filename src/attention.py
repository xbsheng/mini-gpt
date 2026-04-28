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

        # 因果掩码
        mask = torch.triu(torch.ones(attn_scores.shape), diagonal=1)
        print(mask)
        # tensor([[0., 1., 1.],
        #         [0., 0., 1.],
        #         [0., 0., 0.]])

        masked_attn_scores = attn_scores.masked_fill(mask == 1, -torch.inf)
        print(masked_attn_scores)
        # tensor([[ -0.2558,     -inf,     -inf],
        #         [  3.2698, -10.0323,     -inf],
        #         [  2.9594,  -4.3804,  -8.4760]], grad_fn=<MaskedFillBackward0>)

        d_k = keys.shape[-1]
        # 缩放点积注意力(scaled dot-product attention)
        attn_weights = torch.softmax(masked_attn_scores / (d_k**0.5), dim=-1)
        print(attn_weights)
        # tensor([[1.0000, 0.0000, 0.0000],
        #         [0.6966, 0.3034, 0.0000],
        #         [0.4714, 0.2980, 0.2307]], grad_fn=<SoftmaxBackward0>)

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
