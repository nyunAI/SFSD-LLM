from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from datasets import load_dataset
from trainer import LocalTrainer
from layers import DecomposeLinearEigen

device = "cuda"
model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained(
    "t5-small", 
    trust_remote_code=True, 
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("multi_nli", split="validation_matched")
def preprocess_function(sample):
    example = {}
    example['text'] = f"mnli premise: {sample['premise']} hypothesis: {sample['hypothesis']} target:"
    return example

dataset = dataset.map(preprocess_function)

trainer = LocalTrainer(
    model=model,
    max_seq_length=2048,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
)
checkpoint = torch.load('/home/ec2-user/llm_rank/data_aware/mnli_50_query_eigen/checkpoint-28134/pytorch_model.bin')

skip_idxs = []
# for i in range(12):
#     j = -1
#     for layer in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
#         j+=1
#         if f'transformer.h.{i}.attn.attention.{layer}.weight1' in checkpoint:
#             continue
#         else:
#             skip_idxs.append(i*6+j)
#     for layer in ['c_fc', 'c_proj']:
#         j+=1
#         if f'transformer.h.{i}.mlp.{layer}.weight1' in checkpoint:
#             continue
#         else:
#             skip_idxs.append(i*6+j)

# if not f'lm_head.weight1' in checkpoint:
#     skip_idxs.append(72) 

trainer.decomposer_init()
for idx in range(18):#len(trainer.decomposable_layers)):
    if not idx in skip_idxs:
        trainer.decompose_layer(index=idx)

for name, layer in trainer.model.named_modules():
    if isinstance(layer, DecomposeLinearEigen):
        if hasattr(layer, 'init'):
            layer.init = True

trainer.model.load_state_dict(checkpoint)
model = trainer.model
label_map = ["entailment", "neutral", "contradiction"]  # Add more labels as needed
predictions = []
# Forward pass
for sample in tqdm(dataset):
    with torch.no_grad():
        inputs = tokenizer([sample['text']],
                   padding=True,
                   truncation=True,
                   return_tensors="pt").input_ids.to(device)

        logits = model.generate(inputs,
                                do_sample=True,
                                max_length=100
                                )
        prediction = tokenizer.decode(logits[0][1:-1])
        predictions.append(prediction)

# Map the label names to integers for comparison
true_labels = [label_map[example['label']] for example in dataset]

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)

print(f"Accuracy: {accuracy}")
