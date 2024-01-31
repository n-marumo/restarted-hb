This repository provides codes and data for numerical experiments of our paper: https://arxiv.org/abs/2303.01073.


# Execution
To generate numerical results, run [main.py](main.py).
At the bottom of [main.py](main.py), five functions are called:
- `execute_benchmark`
- `execute_rosenbrock`
- `execute_classification_mnist`
- `execute_ae_mnist`
- `execute_mf_movielens`

They correspond to each problem instance.
Before executing [main.py](main.py), comment out some of these function calls if necessary.
The time limit and other parameters can be changed using options.

For the execution of function `execute_mf_movielens`, the MovieLens dataset should be placed in the appropriate folder.
The default relative path is `../dataset/ml-100k/u.data`.
This can be changed by editing the corresponding section of [mf_movielens.py](problem/mf_movielens.py).
The dataset can be downloaded from https://grouplens.org/datasets/movielens/.


# Result
The resulting CSV files of [main.py](main.py) are stored in [result/](result/).


# Plot
To plot the results, run the following:
- [visualizer/compare_methods.py](visualizer/compare_methods.py)
- [visualizer/objlm.py](visualizer/objlm.py)

The resulting PDF files are stored in [result/](result/).


# Software & package information
We ran the code with Python 3.10.13 and the packages listed in [requirements.txt](requirements.txt).
