import argparse
import os


DATASET_CHOICES = ["SST-2", "CoLA", "WNLI", "RTE", "QNLI"]
PRUNING_CHOICES = ["none"]
SUB_CHOICES = ["none"]


def main(args):
    # get yaml file for experiment, pruning strat, and sub_strat
    # load the dataset into some dataloader object
    save_dir = f"{args.task}/prune_{args.pruning_strat}/sub_{args.sub_strat}"
    counter = 0
    while os.isdir(save_dir):
        counter += 1
    save_dir = f"{save_dir}_{counter}"
    os.makedirs(save_dir)
    args.save_dir = save_dir
    return


def load_conf(yaml_path):
    return


def config_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SST-2", choices=DATASET_CHOICES)
    parser.add_argument(
        "--pruning_strat", type="str", choices=PRUNING_CHOICES, default="none"
    )
    parser.add_argument("--sub_strat", type="str", choices=SUB_CHOICES, default="none")

    return parser.parse_args()


if __name__ == "__main__":
    args = config_options()
    main(args)
