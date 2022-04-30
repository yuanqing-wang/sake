import jax
import jax.numpy as jnp
from flax import linen as nn

class ExpNormalSmearing(nn.Module):
    cutoff_lower: float = 0.0
    cutoff_upper: float = 5.0
    num_rbf: float = 50

    def setup(self):
        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)
        means, betas = self._initial_params()
        self.out_features = self.num_rbf

        self.means = self.param(
            "means",
            nn.initializers.constant(means),
            means.shape,
        )

        self.betas = self.param(
            "betas",
            nn.initializers.constant(betas),
            betas.shape,
        )

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = jnp.exp(
            -self.cutoff_upper + self.cutoff_lower
        )
        means = jnp.linspace(start_value, 1, self.num_rbf)
        betas = jnp.array(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def __call__(self, dist):
        return jnp.exp(
            -self.betas
            * (jnp.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
