import torch
from torch import nn

"""
GELU的平滑特性可以在训练过程中带来更好的优化效果，因为它允许模型参数进行更细微的调整。
相比之下，ReLU在零点处有一个尖锐的拐角，有时会使得优化过程更加困难，特别是在深度或复杂的网络结构中。

此外，ReLU对负输入的输出为0，而GELU对负输入会输出一个小的非零值。
这意味着在训练过程中，接收到负输入的神经元仍然可以参与学习，只是贡献程度不如正输入大。
"""


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
