from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset
from trainer import LocalTrainer
from layers import DecomposeLinearEigen
from lm_eval import tasks, evaluator, utils

device = "cuda"
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
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

trainer = LocalTrainer(
    model=model,
    max_seq_length=2048,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
)
checkpoint = torch.load('/home/ec2-user/llm_rank/data_aware/guanaco_50_all_eigen/checkpoint-55383/pytorch_model.bin')

skip_idxs = []
for i in range(12):
    j = -1
    for layer in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
        j+=1
        if f'transformer.h.{i}.attn.attention.{layer}.weight1' in checkpoint:
            continue
        else:
            skip_idxs.append(i*6+j)
    for layer in ['c_fc', 'c_proj']:
        j+=1
        if f'transformer.h.{i}.mlp.{layer}.weight1' in checkpoint:
            continue
        else:
            skip_idxs.append(i*6+j)

if not f'lm_head.weight1' in checkpoint:
    skip_idxs.append(72) 

trainer.decomposer_init()
for idx in range(len(trainer.decomposable_layers)):
    if not idx in skip_idxs:
        trainer.decompose_layer(index=idx)

for name, layer in trainer.model.named_modules():
    if isinstance(layer, DecomposeLinearEigen):
        if hasattr(layer, 'init'):
            layer.init = True

trainer.model.load_state_dict(checkpoint)
model = trainer.model

results = evaluator.simple_evaluate(
        model=model,
        tasks=['truthfulqa_mc'],
        num_fewshot=0,
        batch_size='auto',
        max_batch_size=128,
        device='cuda:0',
        no_cache=True
    )
print(results)
results = evaluator.simple_evaluate(
        model=model,
        tasks=['hellaswag'],
        num_fewshot=10,
        batch_size='auto',
        max_batch_size=128,
        device='cuda:0',
        no_cache=True
    )
print(results)
results = evaluator.simple_evaluate(
        model=model,
        tasks=['arc_challenge'],
        num_fewshot=25,
        batch_size='auto',
        max_batch_size=4,
        device='cuda:0',
        no_cache=True
    )
print(results)
results = evaluator.simple_evaluate(
        model=model,
        tasks=['hendrycksTest-*'],
        num_fewshot=5,
        batch_size='auto',
        max_batch_size=128,
        device='cuda:0',
        no_cache=True
    )
print(results)