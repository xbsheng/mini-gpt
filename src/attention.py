import tiktoken
import torch

from embedding import token_embedding_layer

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

    # 计算第二个元素的注意力权重得分
    query_2 = token_embeddings[1]

    attn_scores_2 = torch.matmul(token_embeddings, query_2)
    # [-12.9429, 238.1973,  19.6215]
    print(attn_scores_2)

    # 归一化
    attn_wrights_2 = torch.softmax(attn_scores_2, dim=-1)
    print(attn_wrights_2)
