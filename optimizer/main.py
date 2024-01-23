import pandas as pd
from . import method, internal
import numpy as np
from scipy import optimize as spopt
import pprint
import time
import os

DIVERGENCE_RATIO = 1e5  # used for checking divergence
SCIPY_TOL = 1e-15


class SmoothNonconvexMin:
    algorithms = {
        "ourrhb": method.OurRestartedHB,
        "ourragd": method.OurRestartedAGD,
        "gd": method.GradientDescent,
        "jnj2018": method.JNJ2018,
        "ll2022": method.LL2022,
    }

    def __init__(self, instance):
        self.instance = instance
        self.alg: method.Base = None
        self.oracle = internal.Oracle(instance)
        self.elapsed_time = None
        self.store_sols = None
        self.results = None
        self.sols = None

    def __calc_obj_gradnorm(self):
        fgs = [
            self.oracle.func_grad(sol["sol"], False) if None in sol.values() else (sol["obj"], sol["grad"])
            for sol in self.alg.solutions
        ]
        fgs = [(f, np.linalg.norm(g)) for f, g in fgs]
        return tuple(map(min, zip(*fgs)))

    def __store_print_result(self, obj, gradnorm, printed):
        result = {
            "iter": self.alg.iter,
            "elapsed_time": self.elapsed_time,
            "obj": obj,
            "gradnorm": gradnorm,
        }
        result |= self.oracle.count
        result |= self.alg.recorded_params
        result = pd.Series(result).to_frame().T
        if printed:
            pprint.pprint(result.to_dict())
        self.results.append(result)
        if self.store_sols:
            self.sols.append(self.alg.solutions[0]["sol"])

    def solve(
        self,
        alg_id,
        alg_param={},
        max_iter=100,
        timeout=20,
        tol_obj=0,
        tol_grad=0,
        print_interval=1,
        store_sols=False,
    ):
        self.store_sols = store_sols
        self.elapsed_time = 0
        self.oracle.reset_count()
        self.results = []
        self.sols = []
        self.alg: method.Base = SmoothNonconvexMin.algorithms[alg_id](alg_param, self.instance.x0, self.oracle)
        obj, gradnorm = self.__calc_obj_gradnorm()
        self.__store_print_result(obj, gradnorm, True)
        obj_init = obj

        for iter in range(1, max_iter + 1):
            start_time = time.perf_counter()
            self.alg.update(self.oracle)
            end_time = time.perf_counter()
            self.elapsed_time += end_time - start_time

            obj, gradnorm = self.__calc_obj_gradnorm()
            self.__store_print_result(obj, gradnorm, iter % print_interval == 0)
            if (
                obj / obj_init >= DIVERGENCE_RATIO
                or self.elapsed_time >= timeout
                or gradnorm <= tol_grad
                or obj <= tol_obj
            ):
                break

    def save_result(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        df = pd.concat(self.results, ignore_index=True)
        df.to_csv(f"{folder}/{filename}.csv", index=False)
        if self.sols:
            np.savetxt(f"{folder}/sols_{filename}.txt", self.sols)


class SmoothNonconvexMinScipy:
    def __init__(self, instance):
        self.instance = instance
        self.oracle = internal.Oracle(instance)
        self.results = None

    def solve(
        self,
        alg_id,
        iter_start=100,
        timeout=20,
        tol_obj=0,
        tol_grad=0,
    ):
        self.results = pd.DataFrame()
        iter = iter_start
        nit = -1

        while True:
            self.oracle.reset_count()
            start_time = time.perf_counter()
            res = spopt.minimize(
                self.oracle.func_grad,
                # self.oracle.func,
                self.instance.x0,
                jac=True,
                method=alg_id,
                tol=SCIPY_TOL,
                options={"maxiter": iter},
                # options={"maxiter": iter, "disp": True},
            )
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print(res.success)
            print(res.message)
            if res.nit == nit:
                break
            nit = res.nit

            obj, grad = self.oracle.func_grad(res.x, False)
            gradnorm = np.linalg.norm(grad)
            df = pd.DataFrame(
                {
                    "iter": [res.nit],
                    "elapsed_time": [elapsed_time],
                    "obj": obj,
                    "gradnorm": gradnorm,
                    "eval": self.oracle.count["eval"],
                    "grad": self.oracle.count["grad"],
                }
            )
            self.results = pd.concat([self.results, df])
            pprint.pprint(self.results)
            if obj <= tol_obj or gradnorm <= tol_grad or elapsed_time > timeout / 2:
                break
            iter *= 2

    def save_result(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        self.results.to_csv(f"{folder}/{filename}.csv", index=False)
