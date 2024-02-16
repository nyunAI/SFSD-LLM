import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import sys
sys.path.append('../')
import numpy as np
import gc
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
import copy
from datasets import load_dataset
from preprocess import get_combination
from preprocess import get_bookcorpus
# from trainer import LocalTrainer
import argparse
from tqdm import tqdm
from layers import ModuleInjection
from lm_eval import evaluator
from evaluator_modified import simple_evaluate_chunk
from evaluator_modified import full_evaluate
from preprocess import *
import json
import time


parser = argparse.ArgumentParser("main")
parser.add_argument("--dataset", type=str, default="hellaswag")
parser.add_argument("--log_path", type=str, default="surgical_logs.txt")
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--delta", type=str, default=0.0)
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--base_model", type=str, default="decomposed_model_mistral_combination.pt")

args = parser.parse_args()
log_name = f"logs_{args.dataset}_mistral_3.csv"
with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Max Compression for Delta : {args.delta}\n"))
    file.write(json.dumps(str(args)))
    file.write("\n")

    
base_model = torch.load(args.base_model)



decomposable_layers_base = []
max_rank = []
for name, l in base_model.named_modules():
    if isinstance(l, nn.Linear):
        max_rank.append(int(l.weight.data.shape[0]*l.weight.data.shape[1]/(l.weight.data.shape[0]+l.weight.data.shape[1])))
        for eligible_layer in args.layers:
            if eligible_layer in name:
                tokens = name.strip().split(".")
                layer = base_model
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]

                decomposable_layers_base.append([layer, tokens[-1]])
                break

      

def evaluate(temp_model, chunk, size = 0.2, reduce = 'loglikelihood_test'):
        results = simple_evaluate_chunk(
            model=temp_model,
            chunk_num=chunk,
            tasks= [args.dataset],
            num_fewshot=0,
            batch_size=4,
            device="cuda:0",
            no_cache=True,
            limit=size,
            reduce=reduce
        )
        if reduce is not None:
            acc = results['results'][args.dataset]['llt']
        else:
            acc = results['results'][args.dataset]['acc_norm']
        params = 0
        for _, param in temp_model.named_parameters():
            params+=param.numel()
        print(acc, params)
        return acc, params

def evaluate_full(temp_model, size = 0.2, reduce = None):
        results = full_evaluate(
            model=temp_model,
            tasks= [args.dataset],
            num_fewshot=0,
            batch_size=4,
            device="cuda:0",
            no_cache=True,
            limit=size,
            reduce=reduce
        )
        if reduce is not None:
            acc = results['results'][args.dataset]['llt']
        else:
            acc = results['results'][args.dataset]['acc_norm']
        params = 0
        for _, param in temp_model.named_parameters():
            params+=param.numel()
        print(acc, params)
        return acc, params


def evaluate_vanilla(temp_model):
        results = evaluator.simple_evaluate(
            model=temp_model,
            tasks= [args.dataset],
            num_fewshot=0,
            batch_size=4,
            device="cuda:0",
            no_cache=True,
            limit=0.3
        )
        acc = results['results'][args.dataset]['acc_norm']
        params = 0
        for _, param in temp_model.named_parameters():
            params+=param.numel()
        print(acc, params)
        return acc, params


new_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
    # load_in_8bit=True,
)
decomposable_layers_new = []

for name, l in new_model.named_modules():
    if isinstance(l, nn.Linear):
        for eligible_layer in args.layers:
            if eligible_layer in name:
                tokens = name.strip().split(".")
                layer = new_model
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]

                decomposable_layers_new.append([layer, tokens[-1]])
                break


baseline_accs = []
for i in range(3):
    base_acc,_  = evaluate(new_model, chunk = i, size = 0.0666, reduce = None)
    baseline_accs.append(base_acc)


old_acc,_ = evaluate(new_model, chunk=0, size=0.2, reduce = None)
entire_acc,_ = evaluate_full(new_model)
acc_30_cal = []
acc_20_cal = []
layer_ind = []
params_ = []

with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Baseline test set acc disjoint {entire_acc} acc on 20% {old_acc} "))
    file.write("\n")
    file.write(json.dumps(f"Chunk 0 {baseline_accs[0]} Chunk 1 {baseline_accs[1]} Chunk 2 {baseline_accs[2]}"))
    file.write("\n")

for index in tqdm(reversed((range(len(decomposable_layers_base)-1)))):
    if(index<28):
        continue

    parent_layer_base, last_token_base = decomposable_layers_base[index]
    layer_base = copy.deepcopy(getattr(parent_layer_base, last_token_base)).cuda().half()
    parent_layer_new, last_token_new = decomposable_layers_new[index]
    layer_old = copy.deepcopy(getattr(parent_layer_new, last_token_new)).cuda().half()
    setattr(parent_layer_new, last_token_new, layer_base)
    layer_new = getattr(parent_layer_new, last_token_new)
    split_rank = []
    search_space = [1] + list((np.arange(0.1, 1.1, 0.1)*max_rank[index]).astype(np.int32))
    print(search_space)
    for i in range(3):
        ind = len(search_space) -1
        if(len(split_rank)>0 and max(split_rank) == search_space[-1]):
            break
        for j in range(len(search_space)):

            rank = search_space[j]

            V = copy.deepcopy(layer_base.V[:, -rank:]).cuda().half()

            layer_new.weight2.data =  V
            layer_new.weight1.data = (
                torch.transpose (V, 1, 0).to(layer_base.weight.device).half() @ layer_base.weight
            ).cuda().half()

            V_prune = copy.deepcopy(layer_base.V[:, :-rank])
            V_prune = V_prune.to(torch.float32)
            layer_base.Y_sub = layer_base.Y_sub.to(torch.float32)
            layer_new.bias.data = layer_base.b1.cuda().half()
            
            temp =  (V_prune @ V_prune.transpose(1,0) @ layer_base.Y_sub.transpose(1,0)).transpose(1,0).cuda().half()
            layer_new.bias.data += temp
            acc,_ = evaluate(new_model, chunk=i, size=0.0666, reduce = None)
            if(acc>=baseline_accs[i] - args.delta):
                ind = j
                with open(args.log_path, "a") as file:
                    file.write(json.dumps(f"Layer index {index} new  {(acc)}  old  {baseline_accs[i]}  chunk {i} and rank {search_space[j]}"))
                    file.write("\n")
                break
        split_rank.append(search_space[ind])   
    final_rank = max(split_rank)
    rank = final_rank

    V = copy.deepcopy(layer_base.V[:, -rank:]).cuda().half()

    layer_new.weight2.data =  V
    layer_new.weight1.data = (
        torch.transpose (V, 1, 0).to(layer_base.weight.device).half() @ layer_base.weight
    ).cuda().half()

    V_prune = copy.deepcopy(layer_base.V[:, :-rank])
    V_prune = V_prune.to(torch.float32)
    layer_base.Y_sub = layer_base.Y_sub.to(torch.float32)
    layer_new.bias.data = layer_base.b1.cuda().half() + (V_prune @ V_prune.transpose(1,0) @ layer_base.Y_sub.transpose(1,0)).transpose(1,0).cuda().half()
    
    acc,_ = evaluate(new_model, chunk=0, size=0.2, reduce = None)
    if(final_rank == search_space[-1] or acc < old_acc - args.delta):
        setattr(parent_layer_new, last_token_new, layer_old)
        del layer_new
        with open(args.log_path, "a") as file:
            file.write(json.dumps(f"Layer index {index}, Unchanged"))
            file.write("\n")
    else:
        layer_new.V = None
        layer_new.Y_sub = None
        layer_new.weight = None
        with open(args.log_path, "a") as file:
            file.write(json.dumps(f"Layer index {index} max compression {final_rank}"))
            file.write("\n")
    

    if((index+1)%7 == 0):
        with open(args.log_path, "a") as file:
            curr_acc,pm = evaluate_full(new_model)
            if(curr_acc>=entire_acc - entire_acc*0.05):
                torch.save(new_model.half(), f"delta_perf_max_comp_{args.dataset}_mistral_3.pt")
                file.write(json.dumps(f"New delta perf checkpoint with {curr_acc} params {pm}"))
            acc,pm = evaluate(new_model, chunk = 0, size = 0.2, reduce = None)
            acc_30_cal.append(curr_acc)
            acc_20_cal.append(acc)
            layer_ind.append(index)
            params_.append(pm)
            p = np.hstack((np.array(layer_ind).reshape((len(layer_ind),1)), np.array(acc_30_cal).reshape((len(layer_ind),1)), np.array(acc_20_cal).reshape((len(layer_ind),1)),np.array(params_).reshape((len(layer_ind),1))))
            print(p)
            p = pd.DataFrame(p, columns=["layer_ind", "acc_30_cal", "acc_20_cal","params"])
            p.to_csv(log_name, index=False)
            file.write(json.dumps(f"Decomposed till {index} 80% disjoint acc {curr_acc} 20% set acc {acc} params {pm}"))
            file.write("\n")
        torch.save(new_model.half(), f"final_max_comp_{args.dataset}_mistral_3.pt")
    torch.cuda.empty_cache()
    gc.collect()
