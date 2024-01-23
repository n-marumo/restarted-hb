import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pathlib


os.chdir("./result")
ALPHA = 0.8


sns.set_theme()
sns.set(font_scale=1.4)
cmap = plt.get_cmap("tab10")
plt.rcParams["text.usetex"] = True

problem_names = [
    "classification_mnist",
    # "ae_mnist",
    "mf_movielens100k",
]
ylabels = {
    "obj": r"$f(x_k)$",
    "L": r"$\ell$",
    "H": r"$h_k$",
}
marker_labels = ["Restart (successful)", "Restart (unsuccessful)"]
M = len(ylabels)
colors = ["black", cmap(0), cmap(1), cmap(2), cmap(3)]
markers = ["o", "x"]
markerwidth = [1, 2]
common_line_option = {
    "alpha": ALPHA,
    "linewidth": 2,
}
common_marker_option = {
    "linewidth": 0,
    "color": "none",
    "markersize": 12,
}
linestyles = [
    "solid",
    "dotted",
    "dashed",
]


def get_instance_paths(problem_name):
    path = pathlib.Path(f"./{problem_name}")
    return [p for p in path.iterdir() if p.is_dir()]


def trim_data(df: pd.DataFrame, maxiter, last):
    if len(df) > maxiter:
        if last:
            df = df[-maxiter:]
        else:
            df = df[:maxiter]
    return df


def make_figure(problem_name, maxiter=1000000, last=False, legend=False):
    for instance_path in get_instance_paths(problem_name):
        if instance_path.name.startswith("_"):
            continue
        # e.g.: instance_path = path to "result/cs/d200_r10_n50_nnz20_xmax1e-01"

        for alg_path in instance_path.glob("ourrhb_*.csv"):
            print(alg_path)
            df = pd.read_csv(str(alg_path))
            df = trim_data(df, maxiter, last)
            fig, ax = plt.subplots(figsize=(7, 4))

            # plot lines
            for i, (yaxis, ylabel) in enumerate(ylabels.items()):
                ax.plot(
                    df["iter"],
                    df[yaxis],
                    label=ylabel,
                    color=colors[i],
                    linestyle=linestyles[i],
                    **common_line_option,
                )

            # plot markers
            k_restart = df[df["iter_inner"] == 0].index.tolist()
            k_suc = [k for k in k_restart if k > df.index[0] and df.loc[k - 1, "L"] > df.loc[k, "L"]]
            k_unsuc = [k for k in k_restart if k > df.index[0] and df.loc[k - 1, "L"] < df.loc[k, "L"]]

            for i, x in enumerate([k_suc, k_unsuc]):
                ax.plot(
                    x,
                    df.loc[x, "obj"],
                    markeredgecolor=colors[i + M],
                    markeredgewidth=markerwidth[i],
                    marker=markers[i],
                    label=marker_labels[i],
                    **common_marker_option,
                )

            # config
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"$f(x_k)$, $\ell$, or $h_k$")
            ax.set_yscale("log")
            if legend:
                ax.legend(loc="lower right")
            # if (problem_name, last) == ("classification_mnist", True):
            #     ax.set_ylim(bottom=1e-12)
            # elif (problem_name, last) == (
            #     "mf_movielens100k",
            #     True,
            # ) and instance_path.name.endswith("200"):
            #     ax.set_ylim(bottom=1e-11)
            # else:
            ax.set_ylim(bottom=1e-11)
            fig.tight_layout(pad=0.1)
            fig.savefig(f"{str(instance_path)}_objlh_maxiter{maxiter}_last{last}.pdf")
            # plt.close()
            fig.show()


def make_legend():
    sns.set(font_scale=1.5)
    dd = [1, 2]  # dummy data
    ncol = M + len(marker_labels)
    figsize = (10.2, 0.46)

    fig = plt.figure("Line plot")
    legend_fig = plt.figure("Legend plot", figsize=figsize)
    ax = fig.add_subplot(111)

    lines = [ax.plot(dd, dd, color=colors[i], linestyle=linestyles[i], **common_line_option)[0] for i in range(M)]
    lines += [
        ax.plot(
            dd,
            dd,
            marker=markers[i],
            markeredgecolor=colors[i + M],
            markeredgewidth=markerwidth[i],
            **common_marker_option,
        )[0]
        for i in range(len(marker_labels))
    ]
    legend_fig.legend(
        lines,
        list(ylabels.values()) + marker_labels,
        loc="center",
        ncol=ncol,
    )

    legend_fig.tight_layout(pad=0.1)
    fig.tight_layout(pad=0.1)
    legend_fig.show()
    fig.show()
    legend_fig.savefig("legend_objlh.pdf")


for problem_name in problem_names:
    make_figure(problem_name, maxiter=500, last=False)
    make_figure(problem_name, maxiter=500, last=True)
make_legend()
