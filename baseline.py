import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import sys
sys.path.append('../')
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
from datasets import load_dataset
from preprocess import get_bookcorpus
from trainer import LocalTrainer
import argparse
from tqdm import tqdm
from layers import ModuleInjection
from lm_eval import evaluator
from preprocess import *
import json
import time

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,
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
    metrics = {}
    results = evaluator.simple_evaluate(
        model=base_model,
        tasks= ["piqa"],
        # tasks= ["hellaswag"],
        num_fewshot=args.shots,
        batch_size="auto",
        max_batch_size=4,
        device="cuda:0",
        no_cache=True,
    )
    metrics = results["results"]
    with open(args.save_path, "a") as file:
        file.write(json.dumps(f"{metrics}"))
        file.write("\n")


parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default="o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj")
parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
parser.add_argument("--budget", default="0.50")
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--dataset", type=str, default="piqa")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--save_path", type=str, default="decomposed_piqa.txt")
parser.add_argument("--shots", type=int, default=0)


args = parser.parse_args()

with open(args.save_path, "a") as file:
    file.write(json.dumps(f"Decomposition Using Piqa, Dataset {args.dataset} Batch Size  {args.batch_size} model {args.model} budget {args.budget}"))
    file.write("\n")

args.layers = args.layers.split(",")
base_model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    token  = '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'
    # load_in_8bit=True,
)

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
dataset, _, _ = get_dataset(args.dataset)
dataset = dataset.map(generate_and_tokenize_prompt)
dataset = dataset.select_columns(["input_ids", "attention_mask"])
dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

#To run on Book Corpora
# data = get_bookcorpus(tokenizer,args.batch_size,64).to("cuda:0")



base_model.eval()
# print(len(decomposable_layers))
for index in tqdm(range(len(decomposable_layers))):
    if(index<35 or index>35+91):  ## Only decompose 13 intermediate layers
        continue

    if(index%1337 == 0):
        evaluate(args,index)

    parent_layer, last_token = decomposable_layers[index]
    
    setattr(
        parent_layer,
        last_token,
        ModuleInjection.make_decomposable(
            getattr(parent_layer, last_token), args.budget, args.algo
        ),
    )

    for _, param in base_model.named_parameters():
        param.requires_grad = False
    base_model.eval()
    # For running on specific Dataset 
    for inputs in dataloader:
        inputs = {k: inputs[k].to(base_model.device) for k in inputs}    
        _ = base_model(**inputs)
        break

    #For running on Book Corpora
    # _ = base_model(data)  
   
evaluate(args)
