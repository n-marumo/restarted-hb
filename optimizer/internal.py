import jax


class Oracle:
    def __init__(self, instance):
        self.__func = instance.func
        self.count = dict.fromkeys(["eval", "grad"], 0)

    def reset_count(self):
        self.count = dict.fromkeys(self.count.keys(), 0)

    def func(self, x, counted=True):
        if counted:
            self.count["eval"] += 1
        return self.__func(x)

    def grad(self, x, counted=True):
        if counted:
            self.count["grad"] += 1
        return jax.grad(self.__func)(x)

    def func_grad(self, x, counted=True):
        if counted:
            self.count["grad"] += 1
        return jax.value_and_grad(self.__func)(x)
