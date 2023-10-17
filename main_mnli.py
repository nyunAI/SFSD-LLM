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
parser.add_argument("--layers", type=str, default='q,k,v')
parser.add_argument("--budget", type=float, default=0.5)
parser.add_argument("--save_name", type=str, default=None)

args = parser.parse_args()
if args.save_name is None:
    args.save_name = f'mnli_{args.budget}_{args.layers}_eigen'

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
dataset = load_dataset("multi_nli", split="train")
# dataset = load_dataset("multi_nli", split="validation_matched")

def preprocess_function(sample):
    example = {}
    example['text'] = f"mnli premise: {sample['premise']} hypothesis: {sample['hypothesis']} target:"
    return example

dataset = dataset.map(preprocess_function).select(range(50000))
# dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=f"{args.save_name}",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    logging_steps=1,
    lr_scheduler_type="constant",
    log_level="debug",
    num_train_epochs=18,
    save_strategy="epoch",
    save_total_limit=10,
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
    kappa_factor=0.5
)
trainer.train()
