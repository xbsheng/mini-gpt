# uv run -m src.demo.res_net


import torch
from torch import nn

from ..gelu import GELU
from ..utils.seed import set_seed


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_size: list[int], use_shortcut=False):
        super().__init__()
        set_seed()

        self.use_shortcut = use_shortcut

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layer_size[output_idx - 1], layer_size[output_idx]),
                    GELU(),
                )
                for output_idx in range(1, len(layer_size))
            ]
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            _x = layer(x)
            x = x + _x if (self.use_shortcut and (x.shape == _x.shape)) else _x

        return x


def print_grad(net: nn.Module, x: torch.Tensor):
    output: torch.Tensor = net(x)

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, output.sum())
    loss.backward()

    for name, params in net.named_parameters():
        if "weight" in name:
            grad = params.grad.abs().mean().item() if params.grad is not None else "--"
            print(f"{name} - {grad}")


if __name__ == "__main__":
    dim = 8
    layer_size = [dim] * 6

    net = ExampleDeepNeuralNetwork(layer_size, use_shortcut=False)

    net_shortcut = ExampleDeepNeuralNetwork(layer_size, use_shortcut=True)

    x = torch.randn(dim)

    print("without shortcut:", "=" * 50)
    print_grad(net, x)

    print("with shortcut:", "=" * 50)
    print_grad(net_shortcut, x)

    """
    without shortcut:
        layers.0.0.weight - 0.0007008572574704885
        layers.1.0.weight - 0.001359405810944736
        layers.2.0.weight - 0.004122217185795307
        layers.3.0.weight - 0.013468404300510883
        layers.4.0.weight - 0.027497971430420876
    
    with shortcut:
        layers.0.0.weight - 0.8736302852630615
        layers.1.0.weight - 1.7628977298736572
        layers.2.0.weight - 1.500881552696228
        layers.3.0.weight - 0.9439252614974976
        layers.4.0.weight - 1.7716870307922363
    """
