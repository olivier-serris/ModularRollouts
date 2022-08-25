import jax.numpy as jnp
import jax
from flax import linen as nn
import chex
from typing import Callable, List, Optional, Sequence
import hydra
import flax


def identity(x):
    return x


def create_MLP(hidden_sizes, out_dim, activation, output_activation=identity):

    if isinstance(activation, str):
        activation = hydra.utils.get_method(activation)

    if isinstance(output_activation, str):
        output_activation = hydra.utils.get_method(output_activation)

    return MLP(hidden_sizes, out_dim, activation, output_activation)


class MLP(nn.Module):

    hidden_sizes: Sequence[int]
    out_dim: int
    activation: Callable = flax.linen.relu
    output_activation: Callable = identity
    model_name: str = "MLP"

    def setup(self) -> None:
        self.layer_sizes = [*self.hidden_sizes, self.out_dim]
        self.layers = [nn.Dense(feat) for feat in self.layer_sizes]

    def __call__(self, x: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.output_activation(self.layers[-1](x))


# test network :
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    in_dim = 3
    batch_size = 10

    mlp = MLP(hidden_sizes=[2, 2], out_dim=1, activation=nn.relu)

    # init parameters :
    x = jax.random.normal(key, (batch_size, in_dim))
    params = mlp.init(rngs=rng, x=x, rng=key)

    print(mlp.apply)

    results = mlp.apply(params, x[0], rng=key)
    print(results)

    results = mlp.apply(params, x, rng=key)
    print(results)

    batch_apply = jax.vmap(
        lambda param, x, rng: mlp.apply({"mlp", param}, x, rng), (None, 0, None)
    )
    results = batch_apply(params, x, rng=key)
    print(results)
