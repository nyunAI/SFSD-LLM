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
from trl import SFTTrainer

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default='lm_head')
parser.add_argument("--decomposed_layers", type=str, default='Attention.q,Attention.k,Attention.v,Attention.o,DenseReluDense.wi,DenseReluDense.wo')
parser.add_argument("--save_name", type=str, default='models/mnli_auto:0.95_Attention.q,Attention.k,Attention.v,Attention.o,DenseReluDense.wi,DenseReluDense.wo_prune-eigen_regress-weights=0.1_sparsity=0.001_finetune=lm_head')
parser.add_argument("--load_name", type=str, default='models/mnli_auto:0.95_Attention.q,Attention.k,Attention.v,Attention.o,DenseReluDense.wi,DenseReluDense.wo_prune-eigen_regress-weights=0.1_sparsity=0.001/checkpoint-300000/pytorch_model.bin')

args = parser.parse_args()

base_model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained(
    "t5-small",
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("multi_nli", split="train")

def preprocess_function(sample):
    example = {}
    example['text'] = f"mnli premise: {sample['premise']} hypothesis: {sample['hypothesis']} target:"
    return example

dataset = dataset.select(range(100000)).map(preprocess_function)
# dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=f"{args.save_name}",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    logging_steps=1,
    lr_scheduler_type="constant",
    log_level="debug",
    num_train_epochs=1,
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=1e-7,
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
    layers=args.decomposed_layers,
    kappa_factor='auto:0.99',
    algo='prune-eigen',
    regress_weights=0,
    sparsity=0
)
checkpoint = torch.load(args.load_name)
trainer.decomposer_init()
for idx in range(len(trainer.decomposable_layers)):
    trainer.decompose_layer(index=idx)

for name, l in trainer.model.named_modules():
    if hasattr(l, 'init'):
        l.init = True
    if hasattr(l, 'pruned'):
        l.pruned = True

trainer.model.load_state_dict(checkpoint)
trainer.layers = args.layers.split(',')
trainer.algo = 'finetune'
trainer.decomposer_init()

trainer.train()