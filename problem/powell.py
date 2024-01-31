import jax
import jax.numpy as jnp
import functools


class Problem:
    def __init__(self, d, sigma_x0, seed=0):
        assert d % 4 == 0
        key = jax.random.PRNGKey(seed)
        self.d = d
        self.x_opt = jnp.zeros(d)
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0

    @functools.partial(jax.jit, static_argnums=(0,))
    def func(self, x):
        return (
            jnp.sum((x[::4] + 10 * x[1::4]) ** 2)
            + jnp.sum(5 * (x[2::4] - x[3::4]) ** 2)
            + jnp.sum((x[1::4] - 2 * x[2::4]) ** 4)
            + jnp.sum(10 * (x[::4] - x[3::4]) ** 4)
        )
