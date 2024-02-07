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
from dataset_ppl import *
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
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--dataset", type=str, default="winogrande")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--log_path", type=str, default="surgical_logs.txt")
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--delta", type=str, default= 0.1)
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--base_model", type=str, default="decomposed_model_mistral_combination.pt")

args = parser.parse_args()
log_name = f"logs_{args.dataset}_mistral.csv"
with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Max Compression for Delta : {args.delta}\n"))
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

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    
    torch_dtype="auto",
)

def param_counter(model):
        params = 0
        for _, param in model.named_parameters():
            params+=param.numel()
        return params

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


test_loaders = []
ppl_3 = []
for i in range(3):
    _,test_loader = get_loaders_chunk('wikitext2', i, 0.066, tokenizer, seq_len=128, batch_size = 8)
    test_loaders.append(test_loader)
    ppl_3.append(llama_eval(new_model, test_loader, 'cuda'))

_,acc_20_loader = get_loaders_chunk('wikitext2', 0, 0,2, tokenizer, seq_len = 128, batch_size = 8 )
_,full_loader = get_loaders_end('wikitext2', tokenizer, seq_len = 128, batch_size = 8)
full_acc = llama_eval(new_model, full_loader, 'cuda')
acc_20 = llama_eval(new_model, acc_20_loader, 'cuda')

acc_30_cal = []
acc_20_cal = []
layer_ind = []
params_ = []

with open(args.log_path, "a") as file:
    file.write(json.dumps(f"Baseline test set acc disjoint {full_acc} acc on 20% {acc_20} "))
    file.write("\n")
    file.write(json.dumps(f"Chunk 0 {ppl_3[0]} Chunk 1 {ppl_3[1]} Chunk 2 {ppl_3[2]}"))
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
            acc = llama_eval(new_model, test_loaders[i], 'cuda')
            if(acc <= ppl_3[i] + ppl_3[i]*args.delta):
                ind = j
                with open(args.log_path, "a") as file:
                    file.write(json.dumps(f"Layer index {index} new  {(acc)}  old  {ppl_3[i]}  chunk {i} and rank {search_space[j]}"))
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
    
    acc = llama_eval(new_model, acc_20_loader, 'cuda')
    if(final_rank == search_space[-1] or acc < acc_20 + acc_20*args.delta):
        # print(f"New acc {acc} vs old acc{old_acc} Performance Drop --> Unchanged")
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
            curr_acc = llama_eval(new_model, full_loader, 'cuda')
            pm = param_counter(new_model)
            # if(curr_acc>= acc_30 - 0.05):
            #     torch.save(new_model.half(),f'delta_perf_specific_{args.dataset}.pt')
            #     file.write(json.dumps(f"New delta perf checkpoint with {curr_acc} params {pm}"))
            if(curr_acc<= full_acc + full_acc*0.20):
                torch.save(new_model.half(), f"delta_perf_max_comp_{args.dataset}_mistral.pt")
                file.write(json.dumps(f"New delta perf checkpoint with {curr_acc} params {pm}"))
            acc = llama_eval(new_model, acc_20_loader, 'cuda')
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
        torch.save(new_model.half(), f"final_max_comp_{args.dataset}_mistral.pt")
    torch.cuda.empty_cache()
    gc.collect()

