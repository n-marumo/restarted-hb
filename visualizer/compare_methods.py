import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pathlib


os.chdir("./result")
ALPHA = 0.6
OFFSET = 0.05


sns.set_theme()
sns.set(font_scale=1.4)
cmap = plt.get_cmap("tab10")
plt.rcParams["text.usetex"] = True


def get_instance_paths(problem_name):
    path = pathlib.Path(f"./{problem_name}")
    return [p for p in path.iterdir() if p.is_dir()]


def trim_data(df: pd.DataFrame, timeout, max_oracle, tol_obj):
    df = df.assign(oracle=df["eval"] + df["grad"])
    idxs = np.where(df["elapsed_time"] > timeout)[0]
    if len(idxs):
        df = df[: idxs[0]]
    idxs = np.where(df["oracle"] > max_oracle)[0]
    if len(idxs):
        df = df[: idxs[0]]
    idxs = np.where(df["obj"] <= tol_obj)[0]
    if len(idxs):
        df = df[: idxs[0] + 1]
    return df


def make_figure(problem_name, line_option, timeout=1e5, max_oracle=10000, xlim_max=-1, tol_obj=0, legend=False):
    xlabels = {
        "elapsed_time": "Time (sec)",
        "oracle": r"\# Oracle calls",
    }
    ylabels = {
        "obj": "Objective function value",
        "gradnorm": "Gradient norm",
    }

    for instance_path in get_instance_paths(problem_name):
        if instance_path.name.startswith("_"):
            continue
        # e.g.: instance_path = path to "result/cs/d200_r10_n50_nnz20_xmax1e-01"
        print(instance_path)
        for xaxis, xlabel in xlabels.items():
            for yaxis, ylabel in ylabels.items():
                fig, ax = plt.subplots(figsize=(7, 4))
                for alg_id, loption in line_option.items():
                    for file in instance_path.glob(f"{alg_id}*.csv"):
                        df = pd.read_csv(str(file))
                        df = trim_data(df, timeout, max_oracle, tol_obj)
                        ax.plot(df[xaxis], df[yaxis], **loption)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_yscale("log")
                if problem_name == "ae_mnist" and yaxis == "obj":
                    ax.set_ylim(top=4e-2)
                if xlim_max > 0:
                    ax.set_xlim(left=-OFFSET * xlim_max, right=xlim_max * (1 + OFFSET))
                if legend:
                    ax.legend()
                fig.tight_layout(pad=0.1)
                fig.savefig(f"{str(instance_path)}_{xaxis}_{yaxis}.pdf")
                plt.close()


def make_legend(line_option, vertical=False, width=10):
    sns.set(font_scale=1.5)
    x = y = [1, 2]
    if vertical:
        ncol = 1
        figsize = (2, 1.5)
    else:
        ncol = len(line_option)
        figsize = (width, 0.45)

    fig = plt.figure("Line plot")
    legend_fig = plt.figure("Legend plot", figsize=figsize)
    ax = fig.add_subplot(111)
    legend_fig.legend(
        [ax.plot(x, y, **option)[0] for option in line_option.values()],
        [option["label"] for option in line_option.values()],
        loc="center",
        ncol=ncol,
    )
    legend_fig.tight_layout(pad=0.1)
    fig.tight_layout(pad=0.1)
    legend_fig.show()
    fig.show()
    legend_fig.savefig(f"legend.pdf")
    plt.close()


line_option = {
    "ourrhb": {
        "label": r"\textbf{Proposed}",
        "color": "black",
        # "alpha": 1,
        "alpha": ALPHA,
        "linewidth": 3,
    },
    "gd": {
        "label": "GD",
        "color": cmap(0),
        "alpha": ALPHA,
        "linewidth": 2,
    },
    "jnj2018": {
        "label": "JNJ2018",
        "color": cmap(1),
        "alpha": ALPHA,
        "linewidth": 2,
        "linestyle": "dashed",
    },
    "ll2022": {
        "label": "LL2022",
        "color": cmap(2),
        "alpha": ALPHA,
        "linewidth": 2,
        "linestyle": "dashdot",
    },
    "ourragd": {
        "label": "MT2022",
        "color": cmap(3),
        "alpha": ALPHA,
        "linewidth": 2,
        "linestyle": "dotted",
    },
    "scipy_L-BFGS-B": {
        "label": "L-BFGS",
        # "markeredgecolor": cmap(4),
        "marker": "o",
        "linewidth": 0,
        "color": cmap(4),
        "markersize": 8,
        # "markeredgewidth": 0,
    },
}

for problem_name in ["dixon_price", "powell", "qing", "rosenbrock"]:
    make_figure(problem_name, line_option, timeout=50, max_oracle=3000)
make_figure("classification_mnist", line_option, max_oracle=5000)
make_figure("ae_mnist", line_option, max_oracle=15000)
make_figure("mf_movielens100k", line_option, max_oracle=3000)
make_legend(line_option, width=12.1)
