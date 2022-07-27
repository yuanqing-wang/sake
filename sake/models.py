import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Union, List
from .layers import DenseSAKELayer, EquivariantGraphConvolutionalLayer

class DenseSAKEModel(nn.Module):
    hidden_features: int
    out_features: int
    depth: int = 4
    activation: Callable=nn.silu
    update: Union[List[bool], bool]=True
    use_semantic_attention: bool = True
    use_euclidean_attention: bool = True
    use_spatial_attention: bool = True


    def setup(self):
        self.embedding_in = nn.Dense(self.hidden_features)
        self.embedding_out = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
            ],
        )

        if isinstance(self.update, bool):
            update = [self.update for _ in range(self.depth)]
        else:
            update = self.update

        for idx in range(self.depth):
            setattr(
                self,
                "d%s" % idx,
                DenseSAKELayer(
                    hidden_features=self.hidden_features,
                    out_features=self.hidden_features,
                    update=update[idx],
                    use_semantic_attention=self.use_semantic_attention,
                    use_euclidean_attention=self.use_euclidean_attention,
                    use_spatial_attention=self.use_spatial_attention,
                ),
            )

        self.layers = [getattr(self, "d%s" % idx) for idx in range(self.depth)]

    def __call__(self, h, x, v=None, mask=None):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x, v = layer(h, x, v, mask=mask)
        h = self.embedding_out(h)
        return h, x, v


class EquivariantGraphNeuralNetwork(nn.Module):
    hidden_features: int
    out_features: int
    depth: int = 4
    activation: Callable=nn.silu
    update: Union[List[bool], bool]=True

    def setup(self):
        self.embedding_in = nn.Dense(self.hidden_features)
        self.embedding_out = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
            ],
        )


        for idx in range(self.depth):
            setattr(
                self,
                "d%s" % idx,
                EquivariantGraphConvolutionalLayer(
                    hidden_features=self.hidden_features,
                    out_features=self.hidden_features,
                    activation=self.activation
                ),
            )

        self.layers = [getattr(self, "d%s" % idx) for idx in range(self.depth)]

    def __call__(self, h, x, v=None, mask=None):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x, v = layer(h, x, v, mask=mask)
        h = self.embedding_out(h)
        return h, x, v
