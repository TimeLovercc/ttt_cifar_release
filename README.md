# Code for EE554: Computer Vision II Course Project

## Members
Huaisheng Zhu, Teng Xiao, Zhimeng Guo

## Steps
1. Install the packages listed in `requirements.txt`
2. Download the two datasets into the same folder:
	- [CIFAR-10-C](https://arxiv.org/abs/1903.12261) (Hendrycks and Dietterich) 
from [this repository](https://github.com/hendrycks/robustness),
which links to [this shared storage](https://zenodo.org/record/2535967#.Xaf8uedKj-Y).
	- [CIFAR-10.1](https://arxiv.org/abs/1806.00451) (Recht et al.) 
from [this repository](https://github.com/modestyachts/CIFAR-10.1).
3. Set the data folder to where the datasets are stored by editing:
	- `--dataroot` argument in `main.py`.
	- `--dataroot` argument in `baseline.py`.
	- `dataroot` variable in `script_test_c10.py`.
4. Run `script.sh` for the main results (TTT-personalized), and `./baseline/script_baseline.sh` for the baseline results.
4. The results are stored in the respective folders in `results/` and `./baseline/results`.
5. Once everything is finished, the results can be compiled and visualized with the following utilities:
	- `show_table.py` parses the results into tables and prints them.
	- `show_plot.py` makes bar plots, and prints the tables in latex format; requires first running `show_table.py`.
	- `show_grad.py` makes the gradient correlation plot.
	- `./baseline/show_table.py` parses the baseline results into tables and prints them.
	- `./baseline/show_plot.py` makes bar plots for baseline results, and prints the tables in latex format; requires first running `show_table.py`.
	- `./baseline/show_grad.py` makes the gradient correlation plot for baselines.
6. The whole training process may take a Tesla V100 several days. So please change the `CUDA_VISIBLE_DEVICES` in `script.sh` and `./baseline/script_baseline.sh` to select a suitable GPU.
