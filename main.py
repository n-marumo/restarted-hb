import optimizer
import problem


PRINT_INTERVAL = 100
ITER_START = 1
D_BENCHMARK = 10**6
SIGMA_BENCHMARK = 1e0
TIMEOUT_BENCHMARK = 1000
MAX_ORACLE_BENCHMARK = 50000

algs_scipy = [
    "L-BFGS-B",
]
benchmark_functions = {
    "dixon_price": problem.dixon_price.Problem,
    "powell": problem.powell.Problem,
    "qing": problem.qing.Problem,
    "rosenbrock": problem.rosenbrock.Problem,
}


def execute(
    instance,
    algs: list[tuple[str, dict]],
    timeout,
    max_oracle,
    tol_obj,
    tol_grad,
    save=False,
    sols=False,
    save_folder="",
):
    # warming
    compmin = optimizer.SmoothNonconvexMin(instance)
    compmin.solve("ourragd", {}, max_iter=2, timeout=1, print_interval=10)

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
            timeout=timeout,
            max_oracle=max_oracle,
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
    max_oracle,
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
            max_oracle=max_oracle,
            tol_obj=tol_obj,
            tol_grad=tol_grad,
        )
        if save:
            compmin.save_result(save_folder, filename)


def execute_benchmark(func_id, algs, timeout, max_oracle, tol_obj, tol_grad, inst_params):
    problem_folder = f"./result/{func_id}"
    d = inst_params["d"]
    sigma_x0 = inst_params["sigma_x0"]

    instance = benchmark_functions[func_id](d=d, sigma_x0=sigma_x0)
    save_folder = f"{problem_folder}/d{d}_sx0{sigma_x0}"
    print(save_folder)
    execute(instance, algs, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)


def execute_classification_mnist(algs, timeout, max_oracle, tol_obj, tol_grad, train_size=10000, layer_size=[32, 16]):
    problem_folder = "./result/classification_mnist"
    seed = 0

    instance = problem.classification_mnist.Problem(
        activation="sigmoid", layer_size=layer_size, train_size=train_size, seed=seed
    )
    save_folder = f"{problem_folder}/ls{'-'.join(map(str, layer_size))}_ts{train_size}_seed{seed}"
    print(save_folder)
    execute(instance, algs, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)


def execute_ae_mnist(algs, timeout, max_oracle, tol_obj, tol_grad, train_size=10000, layer_size=[32, 16, 32]):
    problem_folder = "./result/ae_mnist"
    seed = 0

    instance = problem.ae_mnist.Problem(layer_size=layer_size, train_size=train_size, seed=seed)
    save_folder = f"{problem_folder}/ls{'-'.join(map(str, layer_size))}_ts{train_size}_seed{seed}"
    print(save_folder)
    execute(instance, algs, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)


def execute_mf_movielens(algs, timeout, max_oracle, tol_obj, tol_grad, dim_feature=100):
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
    execute(instance, algs, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)
    execute_scipy(instance, timeout, max_oracle, tol_obj, tol_grad, save=True, save_folder=save_folder)


execute_benchmark(
    "dixon_price",
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e8}),
        ("ll2022", {"L": 1e8}),
    ],
    timeout=TIMEOUT_BENCHMARK,
    max_oracle=MAX_ORACLE_BENCHMARK,
    tol_obj=1e-5,
    tol_grad=1e-5,
    inst_params={
        "d": D_BENCHMARK,
        "sigma_x0": SIGMA_BENCHMARK,
    },
)

execute_benchmark(
    "powell",
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e3}),
        ("ll2022", {"L": 1e3}),
    ],
    timeout=TIMEOUT_BENCHMARK,
    max_oracle=MAX_ORACLE_BENCHMARK,
    tol_obj=1e-5,
    tol_grad=1e-5,
    inst_params={
        "d": D_BENCHMARK,
        "sigma_x0": SIGMA_BENCHMARK,
    },
)

execute_benchmark(
    "qing",
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e7}),
        ("ll2022", {"L": 1e7}),
    ],
    timeout=TIMEOUT_BENCHMARK,
    max_oracle=MAX_ORACLE_BENCHMARK,
    tol_obj=1e-5,
    tol_grad=1e-5,
    inst_params={
        "d": D_BENCHMARK,
        "sigma_x0": SIGMA_BENCHMARK,
    },
)

execute_benchmark(
    "rosenbrock",
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e4}),
        ("ll2022", {"L": 1e4}),
    ],
    timeout=TIMEOUT_BENCHMARK,
    max_oracle=MAX_ORACLE_BENCHMARK,
    tol_obj=1e-5,
    tol_grad=1e-5,
    inst_params={
        "d": D_BENCHMARK,
        "sigma_x0": SIGMA_BENCHMARK,
    },
)

execute_classification_mnist(
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e0}),
        ("ll2022", {"L": 1e0}),
    ],
    timeout=1000,
    max_oracle=5000,
    tol_obj=1e-10,
    tol_grad=1e-10,
    train_size=10000,
)

execute_ae_mnist(
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e-2}),
        ("ll2022", {"L": 1e-2}),
    ],
    timeout=2000,
    max_oracle=15000,
    tol_obj=1e-10,
    tol_grad=1e-10,
    layer_size=[32, 16, 32],
)

execute_mf_movielens(
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e0}),
        ("ll2022", {"L": 1e-1}),
    ],
    timeout=100,
    max_oracle=5000,
    tol_obj=1e-10,
    tol_grad=1e-10,
    dim_feature=100,
)

execute_mf_movielens(
    algs=[
        ("ourrhb", {"L_init": 1e-3, "L_dec": 0.1}),
        ("ourragd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("gd", {"L_init": 1e-3, "L_dec": 0.9}),
        ("jnj2018", {"L": 1e-1}),
        ("ll2022", {"L": 1e-1}),
    ],
    timeout=100,
    max_oracle=5000,
    tol_obj=1e-10,
    tol_grad=1e-10,
    dim_feature=200,
)
