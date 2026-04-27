import tiktoken
import torch

from embedding import EMBEDDING_DIM, token_embedding_layer

# 全局设置：禁用科学计数法
torch.set_printoptions(sci_mode=False)

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = tokenizer.encode("how are you")  # [4919, 389, 345]
    token_embeddings = token_embedding_layer(torch.tensor(inputs))
    print(token_embeddings.shape, token_embeddings)  # torch.Size([3, 256])

    # 测试示例数据
    # token_embeddings = torch.tensor(
    #     [
    #         [0.43, 0.15, 0.89],  # Your     (x^1)
    #         [0.55, 0.87, 0.66],  # journey  (x^2)
    #         [0.57, 0.85, 0.64],  # starts   (x^3)
    #         [0.22, 0.58, 0.33],  # with     (x^4)
    #         [0.77, 0.25, 0.10],  # one      (x^5)
    #         [0.05, 0.80, 0.55],
    #     ]  # step     (x^6)
    # )

    print("计算第二个元素的注意力得分/权重/上下文向量", "=" * 50)
    query_2 = token_embeddings[1]

    attn_scores_2 = torch.matmul(token_embeddings, query_2)
    # [-12.9429, 238.1973,  19.6215]
    print(attn_scores_2)

    # 归一化
    attn_weights_2 = torch.softmax(attn_scores_2 / EMBEDDING_DIM, dim=-1)
    print(attn_weights_2)
    # tensor([0.2082, 0.5553, 0.2365], grad_fn=<SoftmaxBackward0>)

    context_vec_2 = torch.matmul(attn_weights_2, token_embeddings)
    print(context_vec_2.shape)  # torch.Size([256])

    # ====================================================================

    print("计算所有元素的注意力得分/权重/上下文向量", "=" * 50)

    query = token_embeddings  # 3 * 256

    attn_scores = torch.matmul(query, query.T)
    print(attn_scores)
    # tensor([[281.0500, -12.9429,  20.7460],
    #         [-12.9429, 238.1973,  19.6215],
    #         [ 20.7460,  19.6215, 246.4796]], grad_fn=<MmBackward0>)

    # 归一化
    attn_weights = torch.softmax(attn_scores / EMBEDDING_DIM, dim=-1)
    print(attn_weights)
    # tensor([[0.5956, 0.1889, 0.2155],
    #         [0.2082, 0.5553, 0.2365],
    #         [0.2267, 0.2257, 0.5476]], grad_fn=<SoftmaxBackward0>)

    context_vec = torch.matmul(attn_weights, token_embeddings)
    print(context_vec.shape)  # torch.Size([3, 256])
