import jax.numpy as jnp
import jax
from . import internal


class Base:
    def __init__(self):
        self.iter = 0

    @property
    def recorded_params(self):
        return {}

    @property
    def solutions(self):
        return {}

    def update(self, oracle: internal.Oracle):
        pass


class GradientDescent(Base):
    default_params = {
        "L_init": 1e-3,
        "L_inc": 2,
        "L_dec": 0.9,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.x = x0
        self.func_x, self.grad_x = oracle.func_grad(x0)
        self.L = self.params["L_init"]

    @property
    def recorded_params(self):
        return {
            "L": self.L,
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x, "obj": self.func_x, "grad": self.grad_x},
        ]

    def update(self, oracle: internal.Oracle):
        while True:
            y = self.x - self.grad_x / self.L
            func_y, grad_y = oracle.func_grad(y)
            u = y - self.x
            if func_y <= self.func_x + jnp.vdot(self.grad_x, u) + self.L / 2 * jnp.linalg.norm(u) ** 2:
                break
            else:
                self.L *= self.params["L_inc"]
        self.x = y
        self.func_x = func_y
        self.grad_x = grad_y
        self.L *= self.params["L_dec"]
        self.iter += 1


class OurRestartedHB(Base):
    default_params = {
        "L_init": 1e-3,
        "L_inc": 2,
        "L_dec": 0.1,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.L = self.params["L_init"]
        self.init_epoch(x0, *oracle.func_grad(x0))

    def init_epoch(self, x, func, grad):
        self.x = self.x_best = self.x_bar = x
        self.v = jnp.zeros_like(x)
        self.func_x = self.func_bar = self.func_best = self.func_innerinit = func
        self.grad_x = self.grad_bar = self.grad_best = grad
        self.S = 0
        self.H = 0  # estimate of Hölder constant
        self.iter_inner = 0

    @property
    def recorded_params(self):
        return {
            "L": self.L,
            "H": self.H,
            "iter_inner": self.iter_inner,
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x, "obj": self.func_x, "grad": self.grad_x},
            {"sol": self.x_bar, "obj": self.func_bar, "grad": self.grad_bar},
        ]

    def update(self, oracle: internal.Oracle):
        self.iter += 1
        self.iter_inner += 1
        k = self.iter_inner

        # keep old information
        func_old = self.func_x
        grad_old = self.grad_x

        # update x, y, S
        self.v = self.v - grad_old / self.L
        self.x = self.x + self.v
        norm_v = jnp.linalg.norm(self.v)
        self.S += norm_v**2

        # evaluate func and grad at x
        self.func_x, self.grad_x = oracle.func_grad(self.x)
        if self.func_x < self.func_best:
            self.x_best = self.x
            self.func_best = self.func_x
            self.grad_best = self.grad_x

        # check if L is large enough
        if self.func_x - func_old > jnp.vdot(grad_old, self.v) + self.L / 2 * norm_v**2:
            self.init_epoch(self.x_best, self.func_best, self.grad_best)
            self.L *= self.params["L_inc"]
            return

        # compute H
        norm_grad_bar = jnp.linalg.norm(self.grad_bar)
        self.H = max(
            self.H,
            3 / norm_v**2 * (self.func_x - func_old - jnp.vdot(grad_old + self.grad_x, self.v) / 2),
            (8 / (k * self.S)) ** 0.5 * (norm_grad_bar - self.L / k * norm_v),
        )

        # check restart condition
        if k * (k + 1.0) * self.H > 3 / 8 * self.L:
            self.init_epoch(self.x_best, self.func_best, self.grad_best)
            self.L *= self.params["L_dec"]
            return

        # update x_bar
        self.x_bar = (k * self.x_bar + self.x) / (k + 1)
        self.func_bar, self.grad_bar = oracle.func_grad(self.x_bar)
        if self.func_bar < self.func_best:
            self.x_best = self.x_bar
            self.func_best = self.func_bar
            self.grad_best = self.grad_bar


class OurRestartedAGD(Base):
    default_params = {
        "L_init": 1e-3,
        "L_inc": 2,
        "L_dec": 0.9,
        "M_init": 1e-16,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.L = self.params["L_init"]
        self.init_epoch(x0, *oracle.func_grad(x0))

    def init_epoch(self, x, func, grad):
        self.x = self.y = self.y_bar = x
        self.grad_x = self.grad_y = self.grad_innerinit = grad
        self.func_x = self.func_innerinit = func
        self.S = 0
        self.Z = 1
        self.iter_inner = 0
        self.M = self.params["M_init"]

    @property
    def recorded_params(self):
        return {
            "L": self.L,
            "M": self.M,
            "iter_inner": self.iter_inner,
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x, "obj": self.func_x, "grad": self.grad_x},
            {"sol": self.y_bar, "obj": None, "grad": None},
        ]

    def theta(self, k):
        return k / (k + 1)

    def safe_div(self, a, b):
        if a > 0 and b > 0:
            return a / b
        else:
            return 0

    def update(self, oracle: internal.Oracle):
        self.iter += 1
        self.iter_inner += 1
        k = self.iter_inner
        theta_k = self.theta(k)

        # keep old information
        x_old = self.x
        func_old = self.func_x
        grad_old = self.grad_x

        # update x, y, S
        self.x = self.y - self.grad_y / self.L
        self.y = self.x + theta_k * (self.x - x_old)
        self.S += jnp.linalg.norm(self.x - x_old) ** 2

        # evaluate func and grad at x
        self.func_x, self.grad_x = oracle.func_grad(self.x)

        # check if L is large enough
        if self.func_x - self.func_innerinit > -(1 - theta_k) / 2 * self.L * self.S:
            self.init_epoch(x_old, func_old, grad_old)
            self.L *= self.params["L_inc"]
            return

        # evaluate func and grad at y
        func_y, self.grad_y = oracle.func_grad(self.y)

        # update M
        u = self.y - self.x
        norm_v = jnp.linalg.norm(self.x - x_old)
        self.M = max(
            self.M,
            self.safe_div(
                12 * (func_y - self.func_x - (jnp.vdot((self.grad_y + self.grad_x) / 2, u))),
                jnp.linalg.norm(u) ** 3,
            ),
            self.safe_div(
                jnp.linalg.norm(self.grad_y + theta_k * grad_old - (1 + theta_k) * self.grad_x),
                theta_k * norm_v**2,
            ),
        )

        # check restart condition
        if (self.M / self.L) ** 2 * self.S > (1 - theta_k) ** 5:
            self.init_epoch(self.x, self.func_x, self.grad_x)
            self.L *= self.params["L_dec"]
            return

        # update Z, y_bar
        self.y_bar = (self.y + theta_k * self.Z * self.y_bar) / (1 + theta_k * self.Z)
        self.Z = 1 + theta_k * self.Z


# https://proceedings.mlr.press/v162/li22o.html
# Algorithm 2
class LL2022(Base):
    default_params = {
        "L": 1e0,
        "rho": 1e0,
        "eps": 1e-16,
        "B0": 100,
        "c": 2,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)

        # parameters in Theorem 2.2
        self.eta = 1 / (4 * self.params["L"])
        self.B = (self.params["eps"] / self.params["rho"]) ** 0.5
        self.theta = min(4 * (self.params["eps"] * self.params["rho"] * self.eta**2) ** 0.25, 1)
        self.K = round(1 / self.theta)

        self.B0 = self.params["B0"]
        self.init_epoch(x0, oracle.func(x0))

    def init_epoch(self, x, func):
        self.x = self.x_pre = self.y_hat = self.x_innerinit = x
        self.func_innerinit = func
        self.sqsum_diff = 0
        self.k = self.K0 = 0
        self.S1 = self.S2 = jnp.zeros_like(x)
        self.norm_diff_x_K0 = jnp.inf

    @property
    def recorded_params(self):
        return {
            "B": self.B,
            "B0": self.B0,
            "iter_inner": self.k,
            "K": self.K,
            "inv_step": 1 / self.eta,
            "momentum": 1 - self.theta,
        }

    @property
    def solutions(self):
        res = [{"sol": self.x, "obj": None, "grad": None}]
        if self.k >= self.K / 2:
            res.append({"sol": self.y_hat, "obj": None, "grad": None})
        return res

    def update(self, oracle: internal.Oracle):
        # update x, y
        y = self.x + (1 - self.theta) * (self.x - self.x_pre)
        self.x_pre = self.x
        self.x = y - self.eta * oracle.grad(y)
        norm_diff_x = jnp.linalg.norm(self.x - self.x_pre)
        self.sqsum_diff += norm_diff_x**2

        # compute y_hat
        if self.k <= self.K / 2:
            self.S1 = self.S1 + y
            self.K0, self.norm_diff_x_K0 = self.k, norm_diff_x
        else:
            if self.norm_diff_x_K0 < norm_diff_x:
                self.S2 = self.S2 + y
            else:
                self.S1 = self.S1 + self.S2 + y
                self.S2 = jnp.zeros_like(self.S2)
                self.K0, self.norm_diff_x_K0 = self.k, norm_diff_x
        self.y_hat = self.S1 / (self.K0 + 1)
        self.k += 1

        # check restart condition
        if self.k * self.sqsum_diff > max(self.B, self.B0) ** 2 or self.k > self.K:
            func_x = oracle.func(self.x)
            rhs = -min(
                self.params["eps"] ** 1.5 / self.params["rho"] ** 0.5,
                self.params["eps"] * self.params["L"] / self.params["rho"],
            )
            if func_x - self.func_innerinit <= rhs:
                self.init_epoch(self.x, func_x)
            else:
                self.init_epoch(self.x_innerinit, self.func_innerinit)
                self.B0 /= self.params["c"]

        self.iter += 1


# https://proceedings.mlr.press/v75/jin18a
# Algorithm 2
class JNJ2018(Base):
    default_params = {
        "L": 1e0,
        "rho": 1e0,
        "eps": 1e-16,
        "c": 1e0,
        "chi": 1e0,
        "seed": 0,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)

        # parameters in Equation (3)
        self.eta = 1 / (4 * self.params["L"])
        self.kappa = self.params["L"] / (self.params["rho"] * self.params["eps"]) ** 0.5
        self.theta = min(1 / (4 * self.kappa**0.5), 1)
        self.gamma = self.theta**2 / self.eta
        self.s = self.gamma / (4 * self.params["rho"])
        self.J = self.kappa**0.5 * self.params["chi"] * self.params["c"]
        self.r = self.eta * self.params["eps"] * self.params["chi"] ** -5 * self.params["c"] ** -8

        self.x = x0
        self.v = jnp.zeros_like(x0)
        self.no_perturb_count = 0

        self.key = jax.random.PRNGKey(self.params["seed"])

    @property
    def recorded_params(self):
        return {
            "no_perturb_count": self.no_perturb_count,
            "gamma": self.gamma,
            "inv_step": 1 / self.eta,
            "momentum": 1 - self.theta,
            # "J": self.J,
            # "r": self.r,
        }

    @property
    def solutions(self):
        return [{"sol": self.x, "obj": None, "grad": None}]

    def perturb(self, x, r):
        self.no_perturb_count = 0
        d = len(x)
        self.key, subkey = jax.random.split(self.key)
        v = jax.random.normal(subkey, shape=d + 2)
        v /= jnp.linalg.norm(v)
        return x + v[:d] * r

    def negative_curvature_exploitation(self, x, v, s, oracle: internal.Oracle):
        norm_v = jnp.linalg.norm(v)
        if norm_v >= s:
            return x
        else:
            delta = s * v / norm_v
            return min(
                [
                    (oracle.func(x + delta), x + delta),
                    (oracle.func(x - delta), x - delta),
                ]
            )[1]

    def update(self, oracle: internal.Oracle):
        self.no_perturb_count += 1
        if self.no_perturb_count >= self.J and jnp.linalg.norm(oracle.grad(self.x)) <= self.params["eps"]:
            self.x = self.perturb(self.x, self.r)

        # update y
        y = self.x + (1 - self.theta) * self.v
        func_y, grad_y = oracle.func_grad(y)

        rhs = func_y + jnp.vdot(grad_y, self.x - y) - self.gamma / 2 * jnp.linalg.norm(self.x - y) ** 2
        if oracle.func(self.x) >= rhs:
            # AGD
            x_old = self.x
            self.x = y - self.eta * grad_y
            self.v = self.x - x_old
        else:
            # negative curvature exploitation
            self.x = self.negative_curvature_exploitation(self.x, self.v, self.s, oracle)
            self.v = jnp.zeros_like(self.v)

        self.iter += 1
