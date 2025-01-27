import os
from gns_utils import create_experiment_bash_with, create_expr_folder, create_expr_files


def compare_estimation_parameters(
        name: str,
        models: [str],
        Bb_ratio: int,
        reps: int = 10,
        min_size: int = 100,
        max_size: int = 10_000,
        b_step: int = 100,
        **additional_args
):
    assert len(models) > 1, "You must specify at least one model!\n"


    shell_name, csv_name = create_expr_files(name)
    other_args = f"--csv_path {csv_name} " + f" ".join([f"--{k} {v}" for k, v in additional_args.items()])
    expr_count = 0

    for model in models:
        ## Check if model exists
        path = f"{CKPT_DIR}/{model}"
        assert os.path.exists(path), f"Checkpoint {path} does not exist!"

        ## Initialize set sizes (smallest)
        b = min_size
        B = Bb_ratio * b

        while min_size <= b < B <= max_size:
            ## Create experiment with given values
            arguments = f"--model {model} -b {b} -B {B} -r {reps} -acc -nw " + other_args
            create_experiment_bash_with(args=arguments, bash_file=shell_name)
            expr_count += 1
            ## Set new batch size values
            b += b_step
            B = Bb_ratio * b

    print(f"{expr_count} experiments.")
    print(f"Execute in the root-dir using GPUs:")
    print(f"$ bash gns_experiments/experiment_2/{shell_name}")
    print("------------------------------------------\n")


## For Experiment-2
def compare_models_with_time_intervals(
        name: str,
        n_intervals: int,
        models: [str],
        n_steps: int = 1000,
        **additional_args
    ):
    assert n_steps % n_intervals == 0, f"Total diffusion steps ({n_steps}) must be divisible by n_intervals!\n"

    ## Process arguments and create .sh/.csv names
    shell_name, csv_name = create_expr_files(name)
    other_args = f"--csv_path {csv_name} " + f" ".join([f"--{k} {v}" for k, v in additional_args.items()])

    ## Create time intervals
    timesteps = [
        (i, i + (n_steps // n_intervals)) if i == 0 else (i + 1, i + (n_steps // n_intervals))
        for i in range(0, n_steps, n_steps // n_intervals)
    ]

    ## Create experiment shells
    expr_count = 0
    for model in models:
        for (t_min, t_max) in timesteps:
            arguments = f"--model {model} --t_min {t_min} --t_max {t_max} -acc -nw " + other_args
            create_experiment_bash_with(args=arguments, bash_file=shell_name)
            expr_count += 1

    print(f"{expr_count} experiments.")
    print(f"Execute in the root-dir using GPUs:")
    print(f"$ bash gns_experiments/experiment_2/{shell_name}")
    print("------------------------------------------\n")
