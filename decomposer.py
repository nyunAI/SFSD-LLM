import torch
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from preprocess import get_combination
from preprocess import get_bookcorpus
import argparse
from tqdm import tqdm
from layers import ModuleInjection
from lm_eval import evaluator
from preprocess import *
import json
from dataset_ppl import get_wikitext2

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--dataset", type=str, default="piqa")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--log_path", type=str, default="surgical_logs.txt")
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--weights_name", type=str, default="decomposed_model_mistral_combination.pt")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")

args = parser.parse_args()

with open(args.log_path, "a") as file:
    file.write(json.dumps(str(args)))
    file.write("\n")


base_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    # load_in_8bit=True,
)



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

      


tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

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


# To run on Specific Dataset
if args.dataset == 'wikitext2':
    dataset = get_wikitext2(tokenizer, seq_len = args.seq_len )
    dataloader = DataLoader(dataset, batch_size = args.batch_size)

#To run on Commonsense Reasoning Datasets
elif args.dataset != 'combination' and args.dataset != 'bookcorp':
    dataset, _, _ = get_dataset(args.dataset)
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.select_columns(["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)
    print("Done Loading Data")

#To run on Book Corpora
elif args.dataset == 'bookcorp':
    data = get_bookcorpus(tokenizer, args.batch_size, args.seq_len)

#To run on Comb data
elif args.dataset == 'combination':
    dataset, _, _ = get_combination(args.batch_size)
    dataset = dataset.map(generate_and_tokenize_prompt)
    dataset = dataset.select_columns(["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

else:
    print("Dataset Not Supported")
    exit()

for index in tqdm(range(len(decomposable_layers_base))):
    parent_layer, last_token = decomposable_layers_base[index]
    setattr(
        parent_layer,
        last_token,
        ModuleInjection.make_decomposable(
            getattr(parent_layer, last_token), max_rank[index], args.algo
        ),
    )
    
    for _, param in base_model.named_parameters():
        param.requires_grad = False
if(args.dataset == 'wikitext2'):
    for inputs in dataloader:
        _ = base_model(inputs)
        break

elif(args.dataset!='bookcorp'):
    for inputs in dataloader:
        inputs = {k: inputs[k].to(base_model.device) for k in inputs}
        _ = base_model(**inputs)
        break
else:
    _  = base_model(data)

torch.save(base_model.half(),args.weights_name)