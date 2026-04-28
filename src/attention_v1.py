import tiktoken
import torch

from embedding import EMBEDDING_DIM, token_embedding_layer

# 全局设置：禁用科学计数法
torch.set_printoptions(sci_mode=False)


def handle_context_vec_2(embeddings: torch.Tensor):
    print("计算第二个元素的注意力得分/权重/上下文向量", "=" * 50)
    query_2 = embeddings[1]

    attn_scores_2 = torch.matmul(embeddings, query_2)
    # [-12.9429, 238.1973,  19.6215]
    print(attn_scores_2)

    # 归一化
    attn_weights_2 = torch.softmax(attn_scores_2 / EMBEDDING_DIM, dim=-1)
    print(attn_weights_2)
    # tensor([0.2082, 0.5553, 0.2365], grad_fn=<SoftmaxBackward0>)

    context_vec_2 = torch.matmul(attn_weights_2, embeddings)
    print(context_vec_2.shape)  # torch.Size([256])


def handle_context_vec(embeddings: torch.Tensor):
    print("计算所有元素的注意力得分/权重/上下文向量", "=" * 50)

    query = embeddings  # 3 * 256

    attn_scores = torch.matmul(query, query.T)
    print(attn_scores)
    # tensor([[281.0500, -12.9429,  20.7460],
    #         [-12.9429, 238.1973,  19.6215],
    #         [ 20.7460,  19.6215, 246.4796]], grad_fn=<MmBackward0>)

    # 归一化 / 缩放点积注意力(scaled dot-product attention)
    attn_weights = torch.softmax(attn_scores / EMBEDDING_DIM, dim=-1)
    print(attn_weights)
    # tensor([[0.5956, 0.1889, 0.2155],
    #         [0.2082, 0.5553, 0.2365],
    #         [0.2267, 0.2257, 0.5476]], grad_fn=<SoftmaxBackward0>)

    context_vec = torch.matmul(attn_weights, query)
    print(context_vec.shape)  # torch.Size([3, 256])


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = tokenizer.encode("how are you")  # [4919, 389, 345]
    token_embeddings = token_embedding_layer(torch.tensor(inputs))
    print(token_embeddings.shape)  # torch.Size([3, 256])

    handle_context_vec_2(token_embeddings)

    handle_context_vec(token_embeddings)
