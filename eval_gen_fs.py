from lm_eval import tasks, evaluator, utils
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
parser.add_argument("--baseline", type=bool, default=True)
parser.add_argument("--algo", type=str, default='prune-eigen')
parser.add_argument("--regress_weights", type=float, default=0.1)
parser.add_argument("--sparsity", type=float, default=0.001)
parser.add_argument("--dataset", type=str, default='mnli')
args = parser.parse_args()

device = "cuda"
# model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map={"": 0})
# tokenizer = AutoTokenizer.from_pretrained(
#     "t5-small", 
#     trust_remote_code=True, 
#     torch_dtype="auto",
# )
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
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
if not args.baseline:
    if args.load_name is None:
        args.load_name_folder = f'models/{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_sparsity={args.sparsity}/'
        paths = os.listdir(args.load_name_folder)
        idx = 0
        max_ckpt = 0
        for i, path in enumerate(paths):
            if max_ckpt<int(path.split('-')[-1]):
                max_ckpt = int(path.split('-')[-1])
                idx = i
        args.load_name = f'models/{args.dataset}_{args.budget}_{args.layers}_{args.algo}_regress-weights={args.regress_weights}_sparsity={args.sparsity}/{paths[idx]}/pytorch_model.bin'

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

# results = evaluator.simple_evaluate(
#         model=model,
#         tasks=['hellaswag'],
#         num_fewshot=10,
#         batch_size='auto',
#         max_batch_size=16,
#         device='cuda',
#         no_cache=True
#     )
# print(results)
results = evaluator.simple_evaluate(
        model=model,
        tasks=['truthfulqa_mc'],
        num_fewshot=0,
        batch_size='auto',
        max_batch_size=64,
        device='cuda:0',
        no_cache=True
    )
print(results)
# results = evaluator.simple_evaluate(
#         model=model,
#         tasks=['arc_challenge'],
#         num_fewshot=25,
#         batch_size='auto',
#         max_batch_size=4,
#         device='cuda:0',
#         no_cache=True
#     )
# print(results)
# results = evaluator.simple_evaluate(
#         model=model,
#         tasks=['hendrycksTest-*'],
#         num_fewshot=5,
#         batch_size='auto',
#         max_batch_size=128,
#         device='cuda:0',
#         no_cache=True
#     )
# print(results)