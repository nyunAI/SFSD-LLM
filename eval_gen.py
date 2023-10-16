from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset
from trainer import LocalTrainer
from layers import DecomposeLinearEigen
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
checkpoint = torch.load('/home/ec2-user/llm_rank/data_aware/guanaco_50_all_eigen/checkpoint-6153/pytorch_model.bin')

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

trainer.model.load_state_dict(checkpoint, strict = False)
model = trainer.model
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.max_position_embeddings
stride = 2048
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)