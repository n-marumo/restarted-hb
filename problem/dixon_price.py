import jax
import jax.numpy as jnp
import functools


class Problem:
    def __init__(self, d, sigma_x0, seed=0):
        key = jax.random.PRNGKey(seed)
        self.d = d
        # self.x0 = jax.random.normal(key, (d,)) * sigma_x0
        # self.x0 = jnp.ones((d,))

        self.x_opt = jnp.power(2.0, (jnp.power(2.0, -jnp.arange(self.d)) - 1))
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0
        print(self.x0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def func(self, x):
        return (x[0] - 1) ** 2 + jnp.vdot(jnp.arange(2, self.d + 1), (2 * x[1:] ** 2 - x[:-1]) ** 2)
