"""Локальная проверка S3b JAX-ячейки (исправленные скалярные гейты) — 3 итерации на CPU."""
import time

import jax
import jax.numpy as jnp
import numpy as np

d, N, V = 32, 4096, 256
k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(17), 4)
params = {
    "embed": 0.10 * jax.random.normal(k1, (V, d)),
    "t0":    0.02 * jax.random.normal(k2, (N, d)),
    "w_out": 0.05 * jax.random.normal(k3, (d, V)),
    "w":     0.05 * jax.random.normal(k4, (d, d)),
}
fact, noise = [5, 9, 13, 21, 33, 41], [2, 7, 11, 19, 23, 29, 31, 37, 43, 47]
block = np.array(fact + fact + noise)
T = len(block)
big = [((block[i - 1] * 2654435761 + block[i]) % N) for i in range(1, T)]
rows_idx = np.array([0] + big)


def d1_block(params, delta):
    ids = jnp.asarray(block)
    read = params["t0"][jnp.asarray(rows_idx[1:])] + delta[jnp.asarray(rows_idx[1:])]
    h = params["embed"][ids] + jnp.concatenate([jnp.zeros((1, d)), read])
    logits = h[:-1] @ params["w_out"]
    ce = -jax.nn.log_softmax(logits)[jnp.arange(T - 1), ids[1:]].mean()
    targets = jnp.tanh(h[1:] @ params["w"])
    gates = jax.nn.sigmoid(jnp.full((targets.shape[0],), 0.5))  # скаляр на событие (T-1,)
    return ce, targets, gates


@jax.jit
def step_fn(params, delta):
    ce, targets, gates = d1_block(params, delta)
    a = (gates[:, None] * targets).mean(axis=0)  # гейт-скаляр × цель → A (d,)
    w = gates.mean()
    rows_j = jnp.asarray(rows_idx[1:])
    delta = delta.at[rows_j].set((1 - w) * delta[rows_j] + a[None, :])
    return params, delta, ce


delta = jnp.zeros((N, d))
losses, times = [], []
for step in range(3):
    t0 = time.perf_counter()
    params, delta, ce = step_fn(params, delta)
    jax.block_until_ready(ce)
    losses.append(round(float(ce), 4))
    times.append(round(time.perf_counter() - t0, 3))

report = {"steps": 3, "device": str(jax.devices()[0]), "losses": losses, "times": times,
          "loss_finite": all(x == x for x in losses)}
print("LOCAL_JAX_OK:", report)
assert report["loss_finite"] and all(t < 30 for t in times)
