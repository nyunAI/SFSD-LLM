import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    T5ForConditionalGeneration,
)
from datasets import load_dataset
from trainer import LocalTrainer
import argparse

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default='Attention.o')
parser.add_argument("--budget", default='auto')
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--algo", type=str, default='prune-eigen')
parser.add_argument("--regress_weights", type=float, default=0.1)
parser.add_argument("--sparsity", type=float, default=0.01)
parser.add_argument('--dataset', type=str, default = 'mnli')

args = parser.parse_args()
if args.save_name is None:
    args.save_name = f'{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_sparsity={args.sparsity}'

# load the base model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained(
    "t5-small",
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token
# dataset = load_dataset("imdb", split="train")
# dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
# dataset = load_dataset("multi_nli", split="validation_matched")



def preprocess_function_mnli(sample):
    example = {}
    example['text'] = f"mnli premise: {sample['premise']} hypothesis: {sample['hypothesis']} target:"
    return example

def preprocess_function_boolq(sample):
   example = {}
   example['text'] = f"question: {sample['question']} passage: {sample['passage']}  answer:"
   return example

def preprocess_function_sst2(sample):
   example = {}
   example['text'] = f"sst2 sentence: {sample['sentence']} label:"
   return example

def preprocess_function_stsb(sample):
   example = {}
   example['text'] = f"stsb sentence1: {sample['sentence1']} sentence2: {sample['sentence2']} label:"
   return example

if(args.dataset=='mnli'):
  dataset = load_dataset("multi_nli", split="train")
  preprocess_function = preprocess_function_mnli
  ind = range(100000)

elif(args.dataset=="boolq"):
  dataset = load_dataset("boolq", split="train")
  preprocess_function = preprocess_function_boolq
  ind = range(2500)

elif(args.dataset=='sst2'):
   dataset = load_dataset("sst2", split = "train")
   preprocess_function = preprocess_function_sst2
   ind = range(17000)

elif(args.dataset=='stsb'):
   dataset = load_dataset("glue", "stsb", split = "train")
   preprocess_function = preprocess_function_stsb
   ind = range(1400)


dataset = dataset.map(preprocess_function).select(ind)
# dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=f"{args.save_name}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    logging_steps=1,
    lr_scheduler_type="constant",
    log_level="debug",
    num_train_epochs=100,
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=1e-3,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)
trainer = LocalTrainer(
    model=base_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    layers=args.layers,
    kappa_factor=args.budget,
    algo=args.algo,
    regress_weights=args.regress_weights,
    sparsity=args.sparsity
)
trainer.train()
