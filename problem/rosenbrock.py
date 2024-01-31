import jax
import jax.numpy as jnp
import functools


class Problem:
    def __init__(self, d, sigma_x0, seed=0):
        key = jax.random.PRNGKey(seed)
        self.d = d
        self.x_opt = jnp.ones(d)
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0

    @functools.partial(jax.jit, static_argnums=(0,))
    def func(self, x):
        return 100 * jnp.sum((x[1:] - x[:-1] ** 2) ** 2) + jnp.sum((x[:-1] - 1) ** 2)
