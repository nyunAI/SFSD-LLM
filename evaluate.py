import torch
import pandas as pd
import numpy as np
import argparse
from lm_eval import evaluator
from preprocess import *

def evaluate(args):
    base_model = torch.load(args.load_path).half().to(torch.device('cuda'))
    metrics = {}
    results = evaluator.simple_evaluate(
        model=base_model,
        tasks= ['piqa', 'boolq', 'arc_challenge', 'arc_easy', 'winogrande', 'hellaswag'],
        # tasks= ["hellaswag"],
        num_fewshot=args.shots,
        batch_size="auto",
        max_batch_size=8,
        device="cuda:0",
        no_cache=True,
        limit = 0.25,
    )
    metrics = results["results"]
    datasets = list(results['results'].keys())
    acc = []
    acc_norm =[]
    for dataset in datasets:
        acc.append(results['results'][dataset]['acc'])
        if("acc_norm" in results['results'][dataset].keys()):
            acc_norm.append(results['results'][dataset]['acc_norm'])
        else : 
            acc_norm.append(-1)

    datasets.append("Average")
    acc.append(np.mean(np.array(acc)))
    acc_norm.append(-1)
    x = pd.DataFrame({'datasets' : datasets, 'acc' : acc, 'acc_norm' : acc_norm})
    x.to_csv(args.log_path, index = False)
    print("COmplete")

parser = argparse.ArgumentParser("main")

parser.add_argument("--log_path", type=str, default="eval_stats_budget80.csv")
parser.add_argument("--load_path", type=str, default="./compressed_budget80.pt")
parser.add_argument("--shots", type=int, default=0)


args = parser.parse_args()

evaluate(args)