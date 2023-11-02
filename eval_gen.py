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
parser.add_argument("--dataset", type=str, default='mnli')
args = parser.parse_args()

device = "cuda"
model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(
    "t5-small", 
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

encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

max_length = 512#model.config.max_position_embeddings
stride = 512
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