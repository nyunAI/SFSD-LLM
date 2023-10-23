from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from datasets import load_dataset
from trainer import LocalTrainer
from layers import DecomposeLinearEigen, DecomposeLinearEigenPrune, DecomposeLinearSVDPrune, ChannelPrune, DecomposeLinearSVD
import argparse
import os

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default='Attention.q')
parser.add_argument("--budget", type=float, default=0.5)
parser.add_argument("--load_name", type=str, default=None)
parser.add_argument("--baseline", type=bool, default=False)
parser.add_argument("--algo", type=str, default='prune')
parser.add_argument("--regress_weights", type=bool, default=False)
args = parser.parse_args()

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

if not args.baseline:
    if args.load_name is None:
        args.load_name_folder = f'./mnli_{args.budget}_{args.layers}_{args.algo}_regree-weight={args.regress_weights}/'
        paths = os.listdir(args.load_name_folder)
        idx = 0
        max_ckpt = 0
        for i, path in enumerate(paths):
            if max_ckpt<int(path.split('-')[-1]):
                max_ckpt = int(path.split('-')[-1])
                idx = i
        args.load_name = f'./mnli_{args.budget}_{args.layers}_{args.algo}_regree-weight={args.regress_weights}/{paths[idx]}/pytorch_model.bin'

    trainer = LocalTrainer(
        model=model,
        max_seq_length=2048,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        layers=args.layers,
        kappa_factor=args.budget,
        algo=args.algo
    )
    checkpoint = torch.load(args.load_name)

    skip_idxs = []

    trainer.decomposer_init()
    for idx in range(len(trainer.decomposable_layers)):
        if not idx in skip_idxs:
            trainer.decompose_layer(index=idx)

    for name, l in trainer.model.named_modules():
        if isinstance(l, DecomposeLinearEigenPrune) or isinstance(l, DecomposeLinearSVDPrune) or isinstance(l, ChannelPrune) or isinstance(l, DecomposeLinearEigen) or isinstance(l, DecomposeLinearSVD):
            if hasattr(l, 'init'):
                l.init = True
            if hasattr(l, 'pruned'):
                l.pruned = True

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

print(f"args: {args}, Accuracy: {accuracy}")
