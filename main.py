import os
import optimizer
import problem


MAX_ITER = 1000000
PRINT_INTERVAL = 100
ITER_START = 1

algs_scipy = [
    # "L-BFGS-B",
    # "CG",
]


def execute(
    instance,
    algs: list[tuple[str, dict]],
    timeout,
    tol_obj,
    tol_grad,
    save=False,
    sols=False,
    save_folder="",
):
    # warming
    compmin = optimizer.SmoothNonconvexMin(instance)
    compmin.solve("ourragd", {}, max_iter=5, timeout=2, print_interval=10)

    for alg_id, alg_params in algs:
        filename = "_".join([alg_id] + [k + str(v) for k, v in alg_params.items()])
        print("#########################")
        print(f"##### {alg_id}")
        print("#########################")
        print(alg_params)
        print(filename)

        compmin.solve(
            alg_id,
            alg_params,
            max_iter=MAX_ITER,
            timeout=timeout,
            tol_obj=tol_obj,
            tol_grad=tol_grad,
            print_interval=PRINT_INTERVAL,
            store_sols=sols,
        )
        if save:
            compmin.save_result(save_folder, filename)


def execute_scipy(
    instance,
    timeout,
    tol_obj,
    tol_grad,
    save=False,
    save_folder="",
):
    # warming
    compmin = optimizer.SmoothNonconvexMinScipy(instance)
    compmin.solve("L-BFGS-B", iter_start=1, timeout=0.1)

    for alg_id in algs_scipy:
        filename = f"scipy_{alg_id}"
        print("#########################")
        print(f"##### {alg_id}")
        print("#########################")
        print(filename)

        compmin.solve(
            alg_id,
            iter_start=ITER_START,
            timeout=timeout,
            tol_obj=tol_obj,
            tol_grad=tol_grad,
        )
        if save:
            compmin.save_result(save_folder, filename)


def execute_rosenbrock(algs, timeout, tol_obj, tol_grad, inst_params):
    problem_folder = "./result/rosenbrock"
    a = inst_params["a"]
    b = inst_params["b"]
    d = inst_params["d"]
    x0 = inst_params["x0"]

    instance = problem.rosenbrock.Problem(a=a, b=b, d=d, x0=x0)
    save_folder = f"{problem_folder}/d{d}_a{a}_b{b}_x0{x0}"
    print(save_folder)
    execute(instance, algs, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)


def execute_classification_mnist(algs, timeout, tol_obj, tol_grad, train_size=10000, layer_size=[32, 16]):
    problem_folder = "./result/classification_mnist"
    seed = 0

    instance = problem.classification_mnist.Problem(
        activation="sigmoid", layer_size=layer_size, train_size=train_size, seed=seed
    )
    save_folder = f"{problem_folder}/ls{'-'.join(map(str, layer_size))}_ts{train_size}_seed{seed}"
    print(save_folder)
    execute(instance, algs, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)


def execute_ae_mnist(algs, timeout, tol_obj, tol_grad, train_size=10000, layer_size=[32, 16, 32]):
    problem_folder = "./result/ae_mnist"
    seed = 0

    instance = problem.ae_mnist.Problem(layer_size=layer_size, train_size=train_size, seed=seed)
    save_folder = f"{problem_folder}/ls{'-'.join(map(str, layer_size))}_ts{train_size}_seed{seed}"
    print(save_folder)
    execute(instance, algs, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)


def execute_mf_movielens(algs, timeout, tol_obj, tol_grad, dim_feature=100):
    problem_folder = f"./result/mf_movielens100k"
    reg = "quartic"
    reg_param = 1e0
    init = "svd"

    instance = problem.mf_movielens.Problem(
        regularizer=reg,
        init=init,
        dim_feature=dim_feature,
        reg_param=reg_param,
    )
    save_folder = f"{problem_folder}/reg{reg}{reg_param}_init{init}_dimfeat{dim_feature}"
    print(save_folder)
    execute(instance, algs, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, tol_obj, tol_grad, save=True, save_folder=save_folder)


# execute_rosenbrock(
#     algs=[
#         ("ourragd", {"L_dec": 0.9, "L_init": 1e2, "M_init": 1e0}),
#         ("ourragd", {"L_dec": 0.9, "L_init": 1e3, "M_init": 1e0}),
#         ("ourragd", {"L_dec": 0.9, "L_init": 1e4, "M_init": 1e0}),
#         ("ourragd", {"L_dec": 0.9, "L_init": 1e2, "M_init": 1e1}),
#     ],
#     timeout=10,
#     tol_obj=1e-15,
#     tol_grad=1e-15,
#     inst_params={
#         "a": 1,
#         "b": 100,
#         "d": 2,
#         "x0": 0.5,
#     },
# )

# execute_classification_mnist(
#     algs=[
#         ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
#         # ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
#         # ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
#         # ("jnj2018", {"L": 1e0}),
#         # ("ll2022", {"L": 1e0}),
#     ],
#     timeout=100,
#     tol_obj=1e-10,
#     tol_grad=1e-10,
#     train_size=10000,
# )

# execute_ae_mnist(
#     algs=[
#         ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
#         ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
#         ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
#         ("jnj2018", {"L": 1e-2}),
#         ("ll2022", {"L": 1e-2}),
#     ],
#     timeout=3000,
#     tol_obj=1e-10,
#     tol_grad=1e-10,
#     layer_size=[32, 16, 32],
# )

# execute_mf_movielens(
#     algs=[
#         ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
#         # ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
#         # ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
#         # ("jnj2018", {"L": 1e0}),
#         # ("ll2022", {"L": 1e-1}),
#     ],
#     timeout=100,
#     tol_obj=1e-10,
#     tol_grad=1e-10,
#     dim_feature=100,
# )

# execute_mf_movielens(
#     algs=[
#         ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
#         # ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
#         # ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
#         # ("jnj2018", {"L": 1e-1}),
#         # ("ll2022", {"L": 1e-1}),
#     ],
#     timeout=100,
#     tol_obj=1e-10,
#     tol_grad=1e-10,
#     dim_feature=200,
# )
