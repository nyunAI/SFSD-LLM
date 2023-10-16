import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, T5ForConditionalGeneration
from datasets import load_dataset
from trainer import LocalTrainer

# load the base model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-125m",
    # quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_auth_token=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-125m", 
    trust_remote_code=True, 
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token
# dataset = load_dataset("imdb", split="train")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
training_args = TrainingArguments(
    output_dir='guanaco_50_all_eigen',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=1,
    lr_scheduler_type='constant',
    log_level='debug',
    num_train_epochs=73,
    save_strategy='epoch',
    save_total_limit=10,
    learning_rate=1e-4,
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)
trainer = LocalTrainer(
    model=base_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args
)
trainer.train()
