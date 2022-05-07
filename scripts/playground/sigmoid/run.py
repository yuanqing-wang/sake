import jax
import jax.numpy as jnp
import optax
import numpy as onp
import tqdm
from typing import Callable
from sake.utils import ExpNormalSmearing
from sake.layers import ContinuousFilterConvolutionWithConcatenation
from sake.functional import get_x_minus_xt, get_x_minus_xt_norm, get_h_cat_ht
from functools import partial

def double_sigmoid(x):
    return 2.0 * jax.nn.sigmoid(x)

class DenseSAKELayer(nn.Module):
    out_features : int
    hidden_features : int
    activation : Callable = jax.nn.silu
    n_heads : int = 4

    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features)
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        self.velocity_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False,),

                ),
            ],
        )

        self.semantic_attention_mlp = nn.Sequential(
            [
                nn.Dense(self.n_heads),
                partial(nn.leaky_relu, negative_slope=0.2),
            ],
        )

        self.post_norm_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.hidden_features),
                self.activation,
            ]
        )

        self.v_mixing = nn.Dense(1, use_bias=False)

        log_gamma = -jnp.log(jnp.linspace(1.0, 5.0, 4))
        self.log_gamma = self.param(
            "log_gamma",
            nn.initializers.constant(log_gamma),
            log_gamma.shape,
        )

    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, euclidean_attention, mask=None):
        # (batch_size, n, n, n_coefficients)
        # coefficients = self.coefficients_mlp(h_e_mtx)# .unsqueeze(-1)
        coefficients = h_e_mtx

        # (batch_size, n, n, 3)
        # x_minus_xt = x_minus_xt * euclidean_attention.mean(dim=-1, keepdim=True) / (x_minus_xt_norm + 1e-5)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5) # ** 2

        # (batch_size, n, n, coefficients, 3)
        combinations = jnp.expand_dims(x_minus_xt, -2) * jnp.expand_dims(coefficients, -1)

        if mask is not None:
            combinations = combinations * jnp.expand_dims(jnp.expand_dims(mask, -1), -1)

        # (batch_size, n, n, coefficients)
        combinations_sum = combinations.mean(axis=-3)
        combinations_norm = (combinations_sum ** 2).sum(-1)# .pow(0.5)

        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations, combinations

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.unsqueeze(mask, -1)
        h_e = h_e_mtx.sum(axis=-2)
        return h_e

    def node_model(self, h, h_e, h_combinations):
        out = jnp.concatenate([
                h,
                h_e,
                h_combinations,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

    def euclidean_attention(self, x_minus_xt_norm):
        # (batch_size, n, n, 1)
        _x_minus_xt_norm = x_minus_xt_norm + 1e5 * jnp.expand_dims(jnp.eye(
            x_minus_xt_norm.shape[-2],
            x_minus_xt_norm.shape[-2],
        ), -1)

        att = jax.nn.softmax(
            -_x_minus_xt_norm * jnp.exp(self.log_gamma),
            axis=-2,
        )
        return att

    def semantic_attention(self, h_e_mtx):
        # (batch_size, n, n, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)

        # (batch_size, n, n, n_heads)
        # att = att.view(*att.shape[:-1], self.n_heads)
        att = att - 1e5 * jnp.expand_dims(jnp.eye(
            att.shape[-2],
            att.shape[-2],
        ), -1)
        att = jax.nn.softmax(att, axis=-2)
        return att

    def combined_attention(self, x_minus_xt_norm, h_e_mtx):
        euclidean_attention = self.euclidean_attention(x_minus_xt_norm)
        semantic_attention = self.semantic_attention(h_e_mtx)
        combined_attention = jax.nn.softmax(euclidean_attention * semantic_attention, axis=-2)
        return euclidean_attention, semantic_attention, combined_attention

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def __call__(
            self,
            h,
            x,
            v=None,
            mask=None,
        ):

        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)

        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)

        euclidean_attention, semantic_attention, combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx)
        h_e_att = jnp.expand_dims(h_e_mtx, -1) * jnp.expand_dims(combined_attention, -2)
        h_e_att = jnp.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))
        h_combinations, delta_v = self.spatial_attention(h_e_att, x_minus_xt, x_minus_xt_norm, combined_attention, mask=mask)
        delta_v = self.v_mixing(delta_v.swapaxes(-1, -2)).swapaxes(-1, -2).mean(axis=(-2, -3))

        # h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_att, mask=mask)
        h = self.node_model(h, h_e, h_combinations)

        if v is not None:
            v = self.velocity_model(v, h)
        else:
            v = jnp.zeros_like(x)

        v = delta_v + v

        x = x + v

        return h, x, v

class DenseSAKEModel(nn.Module):
    hidden_features: int
    out_features: int
    depth: int = 4
    activation: Callable=nn.silu

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
                DenseSAKELayer(
                    hidden_features=self.hidden_features,
                    out_features=self.hidden_features,
                ),
            )

    def __call__(self, h, x, v=None, mask=None):
        h = self.embedding_in(h)
        for idx in range(self.depth):
            layer = getattr(self, "d%s" % idx)
            h, x, v = layer(h, x, v, mask=mask)
        h = self.embedding_out(h)
        return h, x, v

def run(data_name):
    data = onp.load("%s_dft.npz" % data_name)
    onp.random.seed(2666)
    idxs = onp.random.permutation(len(data['R']))

    x = jnp.array(data['R'][idxs])
    e = jnp.array(data['E'][idxs])
    i = jnp.array(data['z'])
    f = jnp.array(data['F'][idxs])
    i = jax.nn.one_hot(i, i.max())

    batch_size = 4
    n_tr = n_vl = 1000
    x_tr = x[:n_tr]
    e_tr = e[:n_tr]
    f_tr = f[:n_tr]

    x_vl = x[n_tr:n_tr+n_vl]
    e_vl = e[n_tr:n_tr+n_vl]
    f_vl = f[n_tr:n_tr+n_vl]

    x_te = x[n_tr+n_vl:]
    e_te = e[n_tr+n_vl:]
    f_te = f[n_tr+n_vl:]

    n_batches = int(n_tr / batch_size)

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=e_tr.mean(), std=e_tr.std())
    e_tr_mean = e_tr.mean().item()

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=8,
    )

    @jax.jit
    def get_e_pred(params, x):
        i_tr = jnp.broadcast_to(i, (*x.shape[:-1], i.shape[-1]))
        e_pred, _, __ = model.apply(params, i_tr, x)
        e_pred = e_pred.sum(axis=-2)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, x):
        e_pred = get_e_pred(params, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, x: jax.grad(get_e_pred_sum, argnums=(1,))(params, x)[0])

    def loss_fn(params, x, e, f):
        e_pred = get_e_pred(params, x)
        f_pred = get_f_pred(params, x)
        e_loss = jnp.abs(e_pred - e).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * 0.001

    @jax.jit
    def step(state, x, e, f):
        params = state.params
        grads = jax.grad(loss_fn)(params, x, e, f)
        state = state.apply_gradients(grads=grads)
        return state

    @jax.jit
    def epoch(state, x_tr, e_tr, f_tr):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(x_tr.shape[0]))
        x_tr, e_tr, f_tr = x_tr[idxs], e_tr[idxs], f_tr[idxs]

        x_tr = x_tr.reshape(n_batches, batch_size, x_tr.shape[-2], x_tr.shape[-1])
        e_tr = e_tr.reshape(n_batches, batch_size, e_tr.shape[-1])
        f_tr = f_tr.reshape(n_batches, batch_size, f_tr.shape[-2], f_tr.shape[-1])

        def loop_body(idx_batch, state):
            x = x_tr[idx_batch]
            e = e_tr[idx_batch]
            f = f_tr[idx_batch]
            state = step(state, x, e, f)
            return state

        state = jax.lax.fori_loop(0, n_batches, loop_body, state)
        return state

    from functools import partial

    @partial(jax.jit, static_argnums=(6,))
    def many_epochs(state, x_tr, e_tr, f_tr, n=10):
        def loop_body(idx, state):
            state = epoch(state, x_tr, e_tr, f_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    x0 = x_tr[:batch_size]
    i_tr = jnp.broadcast_to(i, (*x0.shape[:-1], i.shape[-1]))
    params = model.init(key, i_tr, x0)
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=500 * n_batches,
        decay_steps=4500 * n_batches,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    for idx_batch in tqdm.tqdm(range(500)):
        import time
        state = many_epochs(state, x_tr, e_tr, f_tr)
        save_checkpoint("_" + data_name, target=state, step=idx_batch)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
