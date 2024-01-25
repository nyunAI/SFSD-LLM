import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
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
    TrainingArguments,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from preprocess import get_combination
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
    base_model = torch.load(args.load_path).to(torch.device('cuda'))

    ppl = PPLMetric(base_model, tokenizer = tokenizer, datasets = ['wikitext2','ptb'], seq_len = 128, batch_size = 4, device = 'cuda' )
    print(ppl)
    results = evaluator.simple_evaluate(
        model=base_model,
        tasks= ['piqa', 'boolq', 'arc_challenge', 'winogrande', 'hellaswag'],
        # tasks= ["hellaswag"],
        num_fewshot=args.shots,
        batch_size="auto",
        max_batch_size=8,
        device="cuda:0",
        no_cache=True,
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
    x.to_csv(args.log_path, index = False)
    print("Complete")

######################################
### PPL evaluation ### 
import torch
import numpy as np
from tqdm import tqdm

from dataset_ppl import get_loaders

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
#############################################

parser = argparse.ArgumentParser("main")

parser.add_argument("--log_path", type=str, default="arc_easy_compressed.csv")
parser.add_argument("--load_path", type=str, default="/home/wolfi/PTC/latest.pt")

parser.add_argument("--shots", type=int, default=0)

tokenizer = AutoTokenizer.from_pretrained(
    "huggyllama/llama-7b",
    trust_remote_code=True,
    torch_dtype="auto",
)
args = parser.parse_args()

evaluate(args)