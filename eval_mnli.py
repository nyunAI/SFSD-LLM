from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from datasets import load_dataset
from trainer import LocalTrainer
from layers import DecomposeLinearEigen, DecomposeLinearEigenPrune, DecomposeLinearSVDPrune, ChannelPrune, DecomposeLinearSVD
import argparse
import os
from scipy import stats

parser = argparse.ArgumentParser("main")
parser.add_argument("--layers", type=str, default='Attention.q,Attention.k,Attention.v,Attention.o,DenseReluDense.wi,DenseReluDense.wo')
parser.add_argument("--budget", default='auto:0.95')
parser.add_argument("--load_name", type=str, default=None)
parser.add_argument("--baseline", type=bool, default=False)
parser.add_argument("--algo", type=str, default='prune-eigen')
parser.add_argument("--regress_weights", type=float, default=0.1)
parser.add_argument("--sparsity", type=float, default=0.001)
parser.add_argument("--dataset", type=str, default='sst2')
args = parser.parse_args()

device = "cuda"
model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(
    "t5-small", 
    trust_remote_code=True, 
    torch_dtype="auto",
)
tokenizer.pad_token = tokenizer.eos_token
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
  dataset = load_dataset("multi_nli", split="validation_matched")
  preprocess_function = preprocess_function_mnli
  label_map = ["entailment", "neutral", "contradiction"]  # Add more labels as needed
  # Map the label names to integers for comparison
  true_labels = [label_map[example['label']] for example in dataset]
  
elif(args.dataset=="boolq"):
  dataset = load_dataset("boolq", split="validation")
  preprocess_function = preprocess_function_boolq
  true_labels = ["True" if example['answer'] else 'False' for example in dataset] 

elif(args.dataset=="sst2"):
  dataset = load_dataset("sst2", split="validation")
  preprocess_function = preprocess_function_sst2
  true_labels = ["positive" if example['label'] == 1 else 'negative' for example in dataset]

elif(args.dataset=='stsb'):
   dataset = load_dataset("glue", "stsb", split = "validation")
   preprocess_function = preprocess_function_stsb
   true_labels = [example['label'] for example in dataset]

dataset = dataset.map(preprocess_function)

if not args.baseline:
    if args.load_name is None:
        args.load_name_folder = f'data_aware/models/{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_sparsity={args.sparsity}/'
        paths = os.listdir(args.load_name_folder)
        idx = 0
        max_ckpt = 0
        for i, path in enumerate(paths):
            if max_ckpt<int(path.split('-')[-1]):
                max_ckpt = int(path.split('-')[-1])
                idx = i
        args.load_name = f'data_aware/models/{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_sparsity={args.sparsity}/{paths[idx]}/pytorch_model.bin'

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

    mask_ckpt = {}
    for key in checkpoint:
        if 'mask' in key:
            mask_ckpt[key] = checkpoint[key]
    trainer.model.load_state_dict(mask_ckpt, strict = False)

    for name, l in trainer.model.named_modules():
        if isinstance(l, DecomposeLinearEigenPrune) or isinstance(l, DecomposeLinearSVDPrune) or isinstance(l, ChannelPrune) or isinstance(l, DecomposeLinearEigen) or isinstance(l, DecomposeLinearSVD):
            if hasattr(l, 'init'):
                l.init = True
            if hasattr(l, 'pruned'):
                l.hard_prune(False)
                
    trainer.model.load_state_dict(checkpoint)
    model = trainer.model
    

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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# Calculate accuracy
if(args.dataset=='stsb'):
    predictions = [float(x) if is_number(x) else 2.5 for x in predictions]
    res = stats.spearmanr(predictions, true_labels)
    print(f"args: {args}, SpearmanCorrelationCoefficient: {res}")
else:
    accuracy = accuracy_score(true_labels, predictions)
    print(f"args: {args}, Accuracy: {accuracy}")


