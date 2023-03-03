from typing import Sequence

import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

jax.config.update("jax_enable_x64", True)

SCALE_IMAGE = 255


# load data
# from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float64))


# MLP
# from https://github.com/google/flax
class MultiLayerPerceptron(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.sigmoid(nn.Dense(feat, dtype=jnp.float64)(x))
        return x


def get_data(train=True, train_size=0, random_state=0):
    data = MNIST("/tmp/mnist/", train=train, download=True, transform=FlattenAndCast())
    images = jnp.array(data.train_data).reshape(len(data.train_data), -1) / SCALE_IMAGE
    labels = jnp.array(data.train_labels)
    if train:
        images, _ = train_test_split(images, stratify=labels, train_size=train_size, random_state=random_state)
    return jnp.array(images)


class Problem:
    def __init__(self, layer_size, train_size, seed=0):
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        random_state = jax.random.randint(subkey, (1,), 0, 2**31)[0].item()
        self.images_train = get_data(train=True, train_size=train_size, random_state=random_state)
        print(self.images_train.shape)
        image_size = self.images_train.shape[1]

        # build model
        self.model = MultiLayerPerceptron(layer_size + [image_size])
        key, subkey = jax.random.split(key)
        tree = self.model.init(key, self.images_train)

        # flatten parameter
        params, self.treedef = jax.tree_util.tree_flatten(tree)
        self.param_shape = [p.shape for p in params]
        self.param_size_cumsum = np.cumsum([p.size for p in params])
        self.x0 = jnp.concatenate([jnp.float64(p.ravel()) for p in params])
        print("Number of variables:", self.param_size_cumsum[-1])

    def unflatten(self, x):
        return jax.tree_util.tree_unflatten(
            self.treedef, [p.reshape(s) for (s, p) in zip(self.param_shape, jnp.split(x, self.param_size_cumsum))]
        )

    def neural_net(self, x, images):
        return self.model.apply(self.unflatten(x), images)

    def inner_func(self, x):
        return self.neural_net(x, self.images_train) - self.images_train

    def outer_func(self, r):
        return jnp.mean(jnp.square(r) / 2)

    @functools.partial(jax.jit, static_argnums=(0,))
    def func(self, x):
        return self.outer_func(self.inner_func(x))
