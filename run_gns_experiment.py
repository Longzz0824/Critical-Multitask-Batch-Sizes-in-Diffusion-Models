import os
import socket
from argparse import ArgumentParser, Namespace
from pathlib import Path


EXPR_DIR = Path("gns_experiments")


def run_experiment_folder(args: Namespace):
    path = EXPR_DIR / args.expr_dir

    assert os.path.exists(path), "Experiment directory doesn't exist!"

    print(f"Experiment: {path}\n")
    contents = os.listdir(path)
    print("Directory Content:\n-------------------------------")
    for c in reversed(contents):
        print(c)
    print("-------------------------------\n")

    for shell in sorted([file for file in contents if file.endswith(".sh")]):
        print(f"RUNNING: {shell.strip('.sh')}\n")
        s_path = path / shell
        os.system(f"bash {s_path}")
        print("Experiment done!\n-------------------------------\n")



if __name__ == "__main__":
    host = socket.gethostname()
    print(f"Host: {host}\n")

    parser = ArgumentParser()
    parser.add_argument("--expr_dir", "-ex", choices=list(os.walk(EXPR_DIR))[0][1], required=True)
    parser.add_argument("--repeat", "-rep", type=int, default=1)
    args = parser.parse_args()

    for _ in range(args.repeat):
        try:
            run_experiment_folder(args)
        except AssertionError as e:
            print(f"FAILED: {e}\n")
            print(f"ABORTING!\n-------------------------------\n")

