import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Union, List
from .layers import (
    DenseSAKELayer,
    EquivariantGraphConvolutionalLayer,
    EquivariantGraphConvolutionalLayerWithSmearing,
)

class SAKEModel(nn.Module):
    hidden_features: int
    out_features: int
    depth: int = 4
    activation: Callable=nn.silu
    update: Union[List[bool], bool]=True
    use_semantic_attention: bool = True
    use_euclidean_attention: bool = True
    use_spatial_attention: bool = True
    n_heads: int=4
    cutoff: Callable=None

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
                    n_heads=self.n_heads,
                    cutoff=self.cutoff,
                ),
            )

        self.layers = [getattr(self, "d%s" % idx) for idx in range(self.depth)]


class DenseSAKEModel(SAKEModel):
    def __call__(self, h, x, v=None, mask=None, he=None):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x, v = layer(h, x, v, mask=mask, he=he)
        h = self.embedding_out(h)
        return h, x, v
    
class SparseSAKEModel(SAKEModel):
    def __call__(self, h, x, v=None, idxs=None):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x, v = layer(h, x, v, idxs=idxs)
        h = self.embedding_out(h)
        return h, x, v


class EquivariantGraphNeuralNetwork(nn.Module):
    hidden_features: int
    out_features: int
    depth: int = 4
    activation: Callable=nn.silu
    update: Union[List[bool], bool]=True
    smear: bool = False
    sigmoid: bool = False

    def setup(self):
        self.embedding_in = nn.Dense(self.hidden_features)
        self.embedding_out = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
            ],
        )

        if self.smear:
            layer = EquivariantGraphConvolutionalLayerWithSmearing
        else:
            layer = EquivariantGraphConvolutionalLayer


        for idx in range(self.depth):
            setattr(
                self,
                "d%s" % idx,
                layer(
                    hidden_features=self.hidden_features,
                    out_features=self.hidden_features,
                    activation=self.activation,
                    update=self.update,
                    sigmoid=self.sigmoid
                ),
            )

        self.layers = [getattr(self, "d%s" % idx) for idx in range(self.depth)]

    def __call__(self, h, x, v=None, mask=None, he=None):
        h = self.embedding_in(h)
        if v is None:
            v = jnp.zeros_like(x)
        for layer in self.layers:
            h, x, v = layer(h, x, v, mask=mask, he=he)
        h = self.embedding_out(h)
        return h, x, v
