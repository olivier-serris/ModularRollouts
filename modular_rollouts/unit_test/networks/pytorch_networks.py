import torch
import torch.nn as nn
import hydra


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_sizes,
        activation,
        output_activation=torch.nn.Identity,
    ) -> None:
        super().__init__()
        layers = []
        layer_sizes = [in_dim, *hidden_sizes, out_dim]
        if isinstance(activation, str):
            activation = hydra.utils.get_method(activation)
        if isinstance(output_activation, str):
            output_activation = hydra.utils.get_method(output_activation)
        for j in range(len(layer_sizes) - 1):
            act = activation if j < len(layer_sizes) - 2 else output_activation
            layers += [nn.Linear(layer_sizes[j], layer_sizes[j + 1]), act()]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class DiscreteSampleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = torch.softmax(x, dim=-1)
        # res = torch.distributions.Categorical(x).sample()
        res = x.argmax(1)
        return res
