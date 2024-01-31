import functools
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import pandas as pd

jax.config.update("jax_enable_x64", True)

DATASET_FOLDER = "../dataset/ml-100k"


def df_to_sparse_matrix(df: pd.DataFrame):
    for cid in ("user", "item"):
        vs = df[cid].unique()
        df[cid] = df[cid].map(dict(zip(vs, range(len(vs)))))
    return {
        "indices": (list(df["user"]), list(df["item"])),
        "values": jnp.array(list(df["rating"])),
    }


def train_test_random(train_frac, key):
    df = pd.read_table(f"{DATASET_FOLDER}/u.data", names=("user", "item", "rating", "time"))
    key, subkey = jax.random.split(key)
    random_state = jax.random.randint(subkey, (1,), 0, 2**31)[0].item()
    df = df.sample(frac=1, random_state=random_state)
    n_total = len(df)
    n_train = int(n_total * train_frac)
    return df_to_sparse_matrix(df[:n_train]), df_to_sparse_matrix(df[n_train:-1])


def train_test_u1():
    df_train = pd.read_table(f"{DATASET_FOLDER}/u1.base", names=("user", "item", "rating", "time"))
    df_test = pd.read_table(f"{DATASET_FOLDER}/u1.test", names=("user", "item", "rating", "time"))
    return df_to_sparse_matrix(df_train), df_to_sparse_matrix(df_test)


def get_traindata_all():
    df_train = pd.read_table(f"{DATASET_FOLDER}/u.data", names=("user", "item", "rating", "time"))
    return df_to_sparse_matrix(df_train)


class Problem:
    def __init__(self, regularizer, init, dim_feature, reg_param, sigma_init=1e-3, seed=0):
        key = jax.random.PRNGKey(seed)
        self.dim_feat = dim_feature
        self.reg_param = reg_param

        self.loss = self.quadratic_loss
        if regularizer == "frobenius":
            self.regularizer = self.frobenius
        elif regularizer == "quartic":
            self.regularizer = self.quartic

        self.data_train = get_traindata_all()

        self.n_train = len(self.data_train["indices"][0])
        self.num_user = max(self.data_train["indices"][0]) + 1
        self.num_item = max(self.data_train["indices"][1]) + 1

        if init == "random":
            params = self.init_random(sigma_init, key)
        elif init == "svd":
            params = self.init_svd()

        self.param_shape = [p.shape for p in params]
        self.param_size_cumsum = np.cumsum([p.size for p in params])
        self.x0 = jnp.concatenate([jnp.float64(p.ravel()) for p in params])

        print(f"#traindata: {self.n_train}")
        print(f"#user: {self.num_user}")
        print(f"#item: {self.num_item}")
        print(f"#variable: {self.param_size_cumsum[-1]}")

    def init_random(self, sigma, key):
        key, *subkeys = jax.random.split(key, num=3)
        return [
            jax.random.normal(subkeys[0], (self.num_user, self.dim_feat)) * sigma,
            jax.random.normal(subkeys[1], (self.num_item, self.dim_feat)) * sigma,
        ]

    def init_svd(self):
        A = coo_matrix(
            (self.data_train["values"], self.data_train["indices"]),
            shape=(self.num_user, self.num_item),
            dtype=np.float64,
        )
        u, s, vT = svds(A, k=self.dim_feat)
        return [u @ np.diag(np.sqrt(s)), vT.T @ np.diag(np.sqrt(s))]

    def unflatten(self, x):
        return [p.reshape(s) for (s, p) in zip(self.param_shape, jnp.split(x, self.param_size_cumsum))]

    def quadratic_loss(self, z):
        return jnp.mean(jnp.square(z)) / 2

    def frobenius(self, u, v):
        return (jnp.sum(jnp.square(u)) + jnp.sum(jnp.square(v))) / 2

    def quartic(self, u, v):
        return jnp.sum(jnp.square(u.T @ u - v.T @ v)) / 2

    def model(self, u, v):
        rows, cols = self.data_train["indices"]
        return (u @ v.T)[(rows, cols)] - self.data_train["values"]

    @functools.partial(jax.jit, static_argnums=(0,))
    def func(self, x):
        u, v = self.unflatten(x)
        return self.loss(self.model(u, v)) + self.regularizer(u, v) * self.reg_param / self.n_train
