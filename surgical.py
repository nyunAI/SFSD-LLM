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
from preprocess import *
import json
import time


parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--dataset", type=str, default="winogrande")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--log_path", type=str, default="surgical_logs.txt")
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
parser.add_argument("--base_model", type=str, default="decomposed_model_arc_easy.pt")

args = parser.parse_args()

with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Specific Dataset : {args.dataset}\n"))
    file.write(json.dumps(str(args)))
    file.write("\n")


# base_model = AutoModelForCausalLM.from_pretrained(
#     "huggyllama/llama-7b",
#     torch_dtype=torch.float32,
#     # device_map="auto",
#     trust_remote_code=True,
#     # load_in_8bit=True,
# ).to(torch.device('cpu'))
    
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
            acc = results['results'][args.dataset]['acc']
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
        acc = results['results'][args.dataset]['acc']
        params = 0
        for _, param in temp_model.named_parameters():
            params+=param.numel()
        print(acc, params)
        return acc, params

# tokenizer = AutoTokenizer.from_pretrained(
#     args.model,
#     trust_remote_code=True,
#     torch_dtype="auto",
# )
# tokenizer.pad_token = tokenizer.eos_token

# data_collator = DataCollatorForSeq2Seq(
#     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
# )

# def tokenize(prompt, add_eos_token=True):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=args.seq_len,
#         padding='max_length',
#         return_tensors=None,
#     )
#     if (
#         result["input_ids"][-1] != tokenizer.eos_token_id
#         and len(result["input_ids"]) < 2048
#         and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)

#     result["labels"] = result["input_ids"].copy()

#     return result


# def generate_and_tokenize_prompt(data_point):
#     full_prompt = data_point["text"]
#     tokenized_full_prompt = tokenize(full_prompt)
#     return tokenized_full_prompt


# # To run on Specific Dataset
# if args.dataset != 'combination' and args.dataset != 'bookcorp':
#     dataset, _, _ = get_dataset(args.dataset)
#     dataset = dataset.map(generate_and_tokenize_prompt)
#     dataset = dataset.select_columns(["input_ids", "attention_mask"])
#     dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)
#     print("Done Loading Data")

# #To run on Book Corpora
# elif args.dataset == 'bookcorp':
#     data = get_bookcorpus(tokenizer,args.batch_size,128)

# #To run on Comb data
# elif args.dataset == 'combination':
#     dataset, _, _ = get_combination(args.batch_size)
#     dataset = dataset.map(generate_and_tokenize_prompt)
#     dataset = dataset.select_columns(["input_ids", "attention_mask"])
#     dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

# else:
#     print("Dataset Not Supported")
#     exit()

# for index in tqdm(range(len(decomposable_layers_base))):
#     if(index<28):
#         continue
#     parent_layer, last_token = decomposable_layers_base[index]
#     setattr(
#         parent_layer,
#         last_token,
#         ModuleInjection.make_decomposable(
#             getattr(parent_layer, last_token), max_rank[index], args.algo
#         ),
#     )
    
#     for _, param in base_model.named_parameters():
#         param.requires_grad = False

# if(args.dataset!='bookcorp'):
#     for inputs in dataloader:
#         print(inputs['input_ids'].shape)
#         inputs = {k: inputs[k].to(base_model.device) for k in inputs}
#         _ = base_model(**inputs)
#         break
# else:
#     _  = base_model(data)


new_model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
    token  = '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'
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
entire_acc,_ = evaluate_vanilla(new_model)
acc_30,_ = evaluate(new_model, chunk = 1, size = 0.3, reduce = None)

with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Baseline test set acc  {entire_acc} acc on 20% {old_acc} acc on disjoint 30% {acc_30}"))
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
            if(acc>=baseline_accs[i]):
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
    if(final_rank == search_space[-1] or acc < old_acc):
        print(f"New acc {acc} vs old acc{old_acc} Performance Drop --> Unchanged")
        setattr(parent_layer_new, last_token_new, layer_old)
        print(new_model)
        del layer_new
        with open(args.log_path, "a") as file:
            file.write(json.dumps(f"Layer index {index}, Unchanged"))
            file.write("\n")
    else:
        # print(new_model)
        # _,_ = evaluate(new_model)
        layer_new.V = None
        layer_new.Y_sub = None
        layer_new.weight = None
        with open(args.log_path, "a") as file:
            file.write(json.dumps(f"Layer index {index} max compression {final_rank}"))
            file.write("\n")
    

    if((index+1)%7 == 0):
        with open(args.log_path, "a") as file:
            curr_acc,pm = evaluate(new_model, chunk = 1, size = 0.3, reduce = None)
            if(curr_acc>= acc_30 - 0.01):
                torch.save(new_model.half(),f'delta_perf_specific_{args.dataset}.pt')
                file.write(json.dumps(f"New delta perf checkpoint with {curr_acc} params {pm}"))
            if(curr_acc>=acc_30):
                torch.save(new_model.half(), f"intact_perf_specific_{args.dataset}.pt")
                file.write(json.dumps(f"New intact perf checkpoint with {curr_acc} params {pm}"))
            acc,pm = evaluate(new_model, chunk = 0, size = 0.2, reduce = None)
            file.write(json.dumps(f"Decomposed till {index} 30% disjoint acc {curr_acc} 20% set acc {acc} params {pm}"))
            file.write("\n")
        torch.save(new_model.half(), f"final_specific_{args.dataset}.pt")
    torch.cuda.empty_cache()
    gc.collect()