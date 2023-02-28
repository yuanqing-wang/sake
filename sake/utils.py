import jax
import jax.numpy as jnp
import numpy as onp
from flax import linen as nn
import math
from typing import Optional

def coloring(x, mean, std):
    return std * x + mean

def cosine_cutoff(x, lower=0.0, upper=5.0):
    cutoffs = 0.5 * (
        jnp.cos(
            math.pi
            * (
                2
                * (x - lower)
                / (upper - lower)
                + 1.0
            )
        )
        + 1.0
    )
    # remove contributions below the cutoff radius
    x = x * (x < upper)
    x = x * (x > lower)
    return cutoffs

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
def mae(x, y):
    return jnp.abs(x - y).mean()

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


def segment_mean(data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False):
  """Returns mean for each segment.
  Args:
    data: the values which are averaged segment-wise.
    segment_ids: indices for the segments.
    num_segments: total number of segments.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether ``segment_ids`` is known to be free of duplicates.
  """
  nominator = jax.ops.segment_sum(
      data,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  denominator = jax.ops.segment_sum(
      jnp.ones_like(data),
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  return nominator / jnp.maximum(denominator,
                                 jnp.ones(shape=[], dtype=denominator.dtype))

def segment_softmax(logits: jnp.ndarray,
                    segment_ids: jnp.ndarray,
                    num_segments: Optional[int] = None,
                    indices_are_sorted: bool = False,
                    unique_indices: bool = False):
  """Computes a segment-wise softmax.
  For a given tree of logits that can be divded into segments, computes a
  softmax over the segments.
    logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
    segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
    segment_softmax(logits, segments)
    >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
    >> dtype=float32)
  Args:
    logits: an array of logits to be segment softmaxed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates
  Returns:
    The segment softmax-ed ``logits``.
  """
  # First, subtract the segment max for numerical stability
  maxs = jax.ops.segment_max(logits, segment_ids, num_segments, indices_are_sorted,
                     unique_indices)
  logits = logits - maxs[segment_ids]
  # Then take the exp
  logits = jnp.exp(logits)
  # Then calculate the normalizers
  normalizers = jax.ops.segment_sum(logits, segment_ids, num_segments,
                            indices_are_sorted, unique_indices)
  normalizers = normalizers[segment_ids]
  softmax = logits / normalizers
  return softmax

def batch(idxs, *hs):
    n_atoms = jnp.array([h.shape[0] for h in hs[0]])
    offsets = jnp.concatenate([jnp.array([0]), n_atoms]).cumsum()
    idxs = jnp.concatenate([idx + offset for (idx, offset) in zip(idxs, offsets)])
    hs = [jnp.concatenate(h) for h in hs]
    return n_atoms, idxs, *hs

def unbatch(n_atoms, idxs, *hs):
    offsets = jnp.concatenate([jnp.array([0]), n_atoms]).cumsum()
    idxs = jnp.split(idxs, n_atoms)
    idxs = jnp.array([idx - offset for (idx, offset) in zip(idxs, offsets)])
    hs = [jnp.split(h, n_atoms) for h in hs]
    return idxs, *hs
    
def batch_and_pad(idxs, *hs, max_n_atoms=None, max_n_edges=None):
    n_atoms = jnp.sum([h.shape[0] for h in hs[0]])
    n_edges = jnp.sum([idx.shape[0] for idx in idxs])
    delta_n_atoms = max_n_atoms - n_atoms
    delta_n_edges = max_n_edges - n_edges
    delta_idxs = jnp.zeros((delta_n_edges, 2))
    delta_hs = [jnp.zeros((delta_n_atoms, *h.shape[1:])) for h in hs]
    return batch(idxs+delta_idxs, *[h + delta_h for (h, delta_h) in zip(hs, delta_hs)])



