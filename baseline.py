import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from lm_eval import evaluator
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from dataset_ppl import get_loaders
from preprocess import get_bookcorpus
import argparse
from tqdm import tqdm
from layers import ModuleInjection
from preprocess import *
import json
import time

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

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

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.seq_len,
        padding='max_length',
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 2048
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point["text"]
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt



def evaluate(args):
    ppl = PPLMetric(base_model, tokenizer = tokenizer, datasets = ['wikitext2','ptb'], seq_len = 128, batch_size = 4, device = 'cuda' )
    print(ppl)
    results = evaluator.simple_evaluate(
        model=base_model,
        tasks= ['piqa', 'boolq', 'arc_challenge', 'arc_easy', 'winogrande', 'hellaswag'],
        # tasks= ["hellaswag"],
        num_fewshot=args.shots,
        batch_size="auto",
        max_batch_size=8,
        device="cuda:0",
        no_cache=True,
        limit = 0.1
    )

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

    datasets.append("Perplexity")
    acc.append(ppl['wikitext2'])
    acc_norm.append(-1)
    datasets.append("Perplexity")
    acc.append(ppl['ptb'])
    acc_norm.append(-1)

    x = pd.DataFrame({'datasets' : datasets, 'acc' : acc, 'acc_norm' : acc_norm})
    x.to_csv(args.save_path, index = False)
    print("Complete")

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
parser.add_argument("--budget", default=0.60)
parser.add_argument("--start_module", type=int, default=31)
parser.add_argument("--algo", type=str, default="svd")
parser.add_argument("--dataset", type=str, default="bookcorp")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--log_path", type=str, default="logs.txt")
parser.add_argument("--save_path", type=str, default="./checksvd_90.pt")
parser.add_argument("--shots", type=int, default=0)


args = parser.parse_args()

with open(args.log_path, "a") as file:
    file.write(json.dumps(str(args)))
    file.write("\n")

args.layers = args.layers.split(",")
base_model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    torch_dtype=torch.float32,
    # device_map="auto",
    trust_remote_code=True,
    # load_in_8bit=True,
).to(torch.device('cpu'))

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

decomposable_layers = []
for name, l in base_model.named_modules():
    if isinstance(l, nn.Linear):
        for eligible_layer in args.layers:
            if eligible_layer in name:
                tokens = name.strip().split(".")
                layer = base_model
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]

                decomposable_layers.append([layer, tokens[-1]])
                
                break
for _, param in base_model.named_parameters():
    param.requires_grad = False

# To run on Specific Dataset
if args.dataset != 'combination' and args.dataset != 'bookcorp':
    dataset, _, _ = get_dataset(args.dataset)
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.select_columns(["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

#To run on Book Corpora
elif args.dataset == 'bookcorp':
    x=1
    # data = get_bookcorpus(tokenizer,512,128)#.to("cuda:0")

#To run on Comb data
elif args.dataset == 'combination':
    dataset, _, _ = get_combination(args.batch_size)
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.select_columns(["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

else:
    print("Dataset Not Supported")
    exit()

base_model.eval()
idx = 0
start = time.time()
for index in tqdm(range(len(decomposable_layers))):
    if(index<7*args.start_module):
        continue
    print(f"Decomposed layer {index} with budget {args.budget}")
    parent_layer, last_token = decomposable_layers[0]
    idx+=1
    args.budget+=0.001
    setattr(
        parent_layer,
        last_token,
        ModuleInjection.make_decomposable(
            getattr(parent_layer, last_token), args.budget, args.algo
        ),
    )

    for _, param in base_model.named_parameters():
        param.requires_grad = False
print(base_model)
print(f"Total layers decomposed {idx}")
exit()
base_model.eval()

# if(args.dataset!='bookcorp'):
#     for inputs in dataloader:
#         print(inputs['input_ids'].shape)
#         inputs = {k: inputs[k].to(base_model.device) for k in inputs}
#         _ = base_model(**inputs)
#         break
# else:
#     _  = base_model(data)
base_model = base_model.half()
# base_model.to('cuda:0')
# evaluate(args)
torch.save(base_model.half(), args.save_path)