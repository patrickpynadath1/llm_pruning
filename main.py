import argparse
import os
import yaml
from utils import *
from pruning import *
from metrics import *
import json
import numpy as np
import random


DATASET_CHOICES = ["sst2", "rte", "cola", "wnli"]
PRUNING_CHOICES = ["none", "var_at_once"]
SUB_CHOICES = ["none", "vanilla_ffn", "svd_ffn", "svd_ffn_train", "sparse_ffn"]


def setup_savedir(args):
    # get yaml file for experiment, pruning strat, and sub_strat
    # load the dataset into some dataloader object
    save_dir = (
        f"{args.output_dir}/{args.task}/prune_{args.pruning_strat}/sub_{args.sub_strat}"
    )
    counter = 0
    while os.path.isdir(f"{save_dir}_{counter}"):
        counter += 1
    save_dir = f"{save_dir}_{counter}"
    os.makedirs(save_dir)
    return save_dir


def load_conf(args):
    yaml_path = args.yaml_path
    task = args.task
    yaml_to_load = []
    yaml_to_load.append(f"{yaml_path}/general.yaml")
    yaml_to_load.append(f"{yaml_path}/{task}.yaml")
    if args.pruning_strat != "none":
        yaml_to_load.append(f"{yaml_path}/prune_strat.yaml")
    if args.sub_strat != "none":
        yaml_to_load.append(f"{yaml_path}/sub_strat.yaml")

    total_conf = {}
    for conf_file in yaml_to_load:
        with open(conf_file, "r") as f:
            conf = yaml.safe_load(f)
            total_conf.update(conf)
    return total_conf


def config_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sst2", choices=DATASET_CHOICES)
    parser.add_argument(
        "--pruning_strat", type=str, choices=PRUNING_CHOICES, default="var_at_once"
    )
    parser.add_argument("--output_dir", type=str, default="outputs_dir")
    parser.add_argument("--yaml_path", type=str, default="yaml_conf")
    parser.add_argument("--sub_strat", type=str, choices=SUB_CHOICES, default="none")

    return parser.parse_args()


def main(args):
    save_dir = setup_savedir(args)
    args.save_dir = save_dir
    conf = load_conf(args)
    # saving the conf for the experiment
    with open(f"{save_dir}/conf.yaml", "w") as f:
        yaml.dump(conf, f)
    tokenizer, model = load_tokenizer_and_model(conf)
    data_df = get_data_df(conf, train=False)
    dataset = GLUE_Dataset(data_df, conf["data_col"], conf["label_col"], tokenizer)
    if args.pruning_strat != "none":
        # select sample indices 
        prune_data_df = get_data_df(conf, train=True)
        prune_dataset = GLUE_Dataset(prune_data_df, conf["data_col"], conf["label_col"], tokenizer)
        num_samples = conf['num_samples']
        sample_indices = list(np.random.choice(len(prune_data_df), num_samples, replace=False))
        # prune
        model = prune_attention_heads(model, 
                                      prune_dataset, 
                                      sample_indices,
                                      save_dir,
                                      conf["topk"])
    if args.sub_strat != "none":
        # v
        model = replace_bert_ffn(model, 
                                 args.sub_strat, 
                                 train_iter=conf[args.sub_strat]["train_iter"], 
                                 save_dir=save_dir,
                                 inner_rank=conf[args.sub_strat]["inner_rank"],
                                 percentile=conf[args.sub_strat]["percentile"],
                                 svd_bias=conf[args.sub_strat]["svd_bias"])
    final_metrics = eval_model(model, dataset)
    # outputting the final eval metrics
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(final_metrics, f)
    return


if __name__ == "__main__":
    args = config_options()
    main(args)
