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
from preprocess import *

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default="query_key_value,dense")
parser.add_argument("--model", type=str, default="EleutherAI/pythia-1b")
parser.add_argument("--budget", default="0.90")
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--algo", type=str, default="eigen")
parser.add_argument("--regress_weights", type=float, default=0.1)
parser.add_argument("--sparsity", type=float, default=0.01)
parser.add_argument("--dataset", type=str, default="winogrande")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--save_strategy", type=str, default="no")
parser.add_argument("--multigpu", type=bool, default=True)
parser.add_argument("--clm", type=bool, default=False)

args = parser.parse_args()
if args.save_name is None:
    if args.clm:
        args.save_name = f"models/global_{args.model}_{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_learning_rate={args.learning_rate}"
    else:
        args.save_name = f"models/local_{args.model}_{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_learning_rate={args.learning_rate}"

# load the base model in 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
if "t5" in args.model:
    base_model = T5ForConditionalGeneration.from_pretrained(
        args.model, device_map={"": 0}
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
        use_auth_token=True,
        # load_in_8bit=True,
    )

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token

dataset, dataset_eval, true_labels = get_dataset(args.dataset)

training_args = TrainingArguments(
    output_dir=f"{args.save_name}",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    lr_scheduler_type="constant",
    log_level="debug",
    num_train_epochs=100,
    save_strategy=args.save_strategy,
    save_total_limit=1,
    learning_rate=args.learning_rate,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    evaluation_strategy="epoch",
)
trainer = LocalTrainer(
    model=base_model,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    layers=args.layers,
    kappa_factor=args.budget,
    algo=args.algo,
    regress_weights=args.regress_weights,
    sparsity=args.sparsity,
    true_labels=true_labels,
    eval_dataset_name=args.dataset,
    model_name=args.model,
    multigpu=args.multigpu,
    eval_epochs=8,
    clm=args.clm,
)
trainer.train()
print(trainer.custom_evaluate(size=None))
