import jax
import jax.numpy as jnp
import numpy as onp
from flax import linen as nn

def coloring(x, mean, std):
    return std * x + mean

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

@jax.jit
def _mae(x, y):
    return jnp.abs(x - y).mean()

@jax.jit
def mae(x, y):
    z = jnp.stack([x, y], axis=1)
    row_mae = lambda z: _mae(z[0], z[1])
    return jax.lax.map(row_mae, z).mean()

@jax.jit
def mae_with_replacement(x, y, seed=0):
    key = jax.random.PRNGKey(seed)
    idxs = jax.random.choice(
        key, x.shape[0], shape=(x.shape[0],), replace=True,
    )
    x = x[idxs]
    y = y[idxs]
    return mae(x, y)

def bootstrap_mae(x, y, n_samples=10, ci=0.95):
    original = jnp.abs(x - y).mean().item()
    results = []
    for idx in range(n_samples):
        result = mae_with_replacement(x, y, idx).item()
        results.append(result)
    low = onp.percentile(results, 100.0 * 0.5 * (1 - ci))
    high = onp.percentile(results, (1 - ((1 - ci) * 0.5)) * 100.0)
    return original, low, high
