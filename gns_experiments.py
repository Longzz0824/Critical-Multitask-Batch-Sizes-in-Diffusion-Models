import os
from pathlib import Path
from gns_utils import create_experiment_bash_with, prepare_expr_files


CKPT_DIR = Path("checkpoints")


#######################################################
#### Experiment-1: Compare true_grad data portions ####
#######################################################
def compare_true_portions(
        name: str,
        expr_dir: str,
        model: str,
        tp: float,
        reps: int
    ):
    assert 0. < tp <= 1., "Data portion must be between 0 and 1."
    ## Process arguments and create .sh/.csv names
    shell_name, csv_name, vis_dir = prepare_expr_files(expr_name=name, expr_dir=expr_dir)

    ## Create experiment shells
    args = f"-p {tp} -r {reps}"
    create_experiment_bash_with(args, model, bash_path=shell_name, csv_path=csv_name, vis_dir=vis_dir)


def experiment_1(models: [str], portions: [float]):
    assert len(models) > 1, "You must specify at least one model!"
    expr_count = 0
    for model in models:
        ## Check if model exists
        path = f"{CKPT_DIR}/{model}"
        assert os.path.exists(path), f"Checkpoint {path} does not exist!"

        for tp in portions:
            expr_name = f"model_{model.strip('.pt')}_tp_0{int(tp*10)}"
            compare_true_portions(name=expr_name,
                                  expr_dir="1_true_grad_accuracy",
                                  model=model,
                                  tp=tp,
                                  reps=5)
            expr_count += 1
    print(f"{expr_count} experiments.")
    print(f"Execute in the root-dir using GPUs:")
    print("------------------------------------------\n")

    print("Done!\n")


#######################################################
######### Experiment-2: Hyperparameter Search ######### TODO: alter method
#######################################################
def compare_estimation_parameters(
        name: str,
        expr_dir: str,
        models: [str],
        Bb_ratio: int,
        reps: int,
        min_size: int,
        max_size: int,
        b_step: int,
    ):
    assert len(models) > 1, "You must specify at least one model!"
    ## Process arguments and create .sh/.csv names
    shell_name, csv_name, vis_dir = prepare_expr_files(expr_name=name, expr_dir=expr_dir)

    ## Create experiment shells
    expr_count = 0
    for model in models:
        ## Check if model exists
        path = f"{CKPT_DIR}/{model}"
        assert os.path.exists(path), f"Checkpoint {path} does not exist!"

        ## Initialize set sizes (smallest)
        b = min_size
        B = Bb_ratio * b

        ## Iterate over batch size values
        while min_size <= b < B <= max_size:
            ## Create experiment with given values
            args = f"-b {b} -B {B} -r {reps}"
            create_experiment_bash_with(args, model, bash_path=shell_name, csv_path=csv_name, vis_dir=vis_dir)
            expr_count += 1
            ## Set new batch size values
            b += b_step
            B = Bb_ratio * b

    print(f"{expr_count} experiments.")
    print(f"Execute in the root-dir using GPUs:")
    print("------------------------------------------\n")


def experiment_2(models: [str], Bb_ratios: [float], reps: [int]):
    for ratio in Bb_ratios:
        for r in reps:
            expr_name = f"all_models_Bb_{ratio}_reps_{r}"
            compare_estimation_parameters(name=expr_name,
                                          expr_dir="2_hyperparameters",
                                          models=models,
                                          Bb_ratio=ratio,
                                          reps=r,
                                          min_size=100,
                                          max_size=10_000,
                                          b_step=100)


#######################################################
########## Experiment-3: Time-step Intervals ##########
#######################################################

def compare_models_with_time_intervals(
        name: str,
        expr_dir: str,
        n_intervals: int,
        models: [str],
        n_steps: int = 1000,
    ):
    assert len(models) > 1, "You must specify at least one model!"
    assert n_steps % n_intervals == 0, f"Total diffusion steps ({n_steps}) must be divisible by n_intervals!"
    ## Process arguments and create .sh/.csv names
    shell_name, csv_name, vis_dir = prepare_expr_files(expr_name=name, expr_dir=expr_dir)

    ## Create time intervals
    timesteps = [
        (i, i + (n_steps // n_intervals)) if i == 0 else (i + 1, i + (n_steps // n_intervals))
        for i in range(0, n_steps, n_steps // n_intervals)
    ]

    ## Create experiment shells
    expr_count = 0
    for model in models:
        ## Check if model exists
        path = f"{CKPT_DIR}/{model}"
        assert os.path.exists(path), f"Checkpoint {path} does not exist!"

        for (t_min, t_max) in timesteps:
            args = f"--t_min {t_min} --t_max {t_max}"
            create_experiment_bash_with(args, model, csv_path=csv_name, bash_path=shell_name, vis_dir=vis_dir)
            expr_count += 1

    print(f"{expr_count} experiments.")
    print(f"Execute in the root-dir using GPUs:")
    print("------------------------------------------\n")


def experiment_3(models: [str], intervals: [int]):
    for i in intervals:
        expr_name = f"models_{i}_intervals"
        compare_models_with_time_intervals(name=expr_name,
                                           expr_dir="3_time_intervals",
                                           n_intervals=i,
                                           models=models)


if __name__ == "__main__":
    ## Get available models (DiT-S/2 checkpoints)
    all_models = sorted(os.listdir(CKPT_DIR))
    best_model = all_models[-1:]
    worst_model = all_models[:1]

    ## Experiment 1: Effect of true_portion
    experiment_1(models=best_model + worst_model, portions=(0.1, 0.2, 0.5))

    ## Experiment 2: Hyperparameter Search
    #experiment_2(models=best_model + worst_model, Bb_ratios=(10, 100), reps=(2, 5))

    ## Experiment 3: Time Intervals
    #experiment_3(models=all_models, intervals=(2, 5, 10, 20))
