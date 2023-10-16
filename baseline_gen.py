import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trainer import LocalTrainer
from trl import SFTTrainer

# load the base model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
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
    output_dir='guanaco_90_qkv_ft',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    log_level='debug',
    num_train_epochs=10,
    save_strategy='epoch',
    save_total_limit=10,
    learning_rate=1e-5,
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)
trainer = LocalTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args
)

trainer.decomposer_init()
for idx in range(len(trainer.decomposable_layers)):
    trainer.decompose_layer(index=idx)

checkpoint = torch.load('/home/ec2-user/llm_rank/data_aware/guanaco_90_qkv/checkpoint-45510/pytorch_model.bin')
trainer.model.load_state_dict(checkpoint)
model = trainer.model

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()
