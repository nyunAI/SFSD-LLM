# Surgical Feature-Space Decomposition of LLMs: Why, When and How?
This repository contains the code for our paper: [Surgical Feature-Space Decomposition of LLMs: Why, When and How?](https://www.arxiv.org/pdf/2405.13039). The paper was published in Association for Computational Linguistics (ACL), [2024] by [Arnav Chavan](https://sites.google.com/view/arnavchavan/), [Nahush Lele](https://www.linkedin.com/in/nahush-lele-a06826204/), and [Deepak Gupta](https://dkgupta90.github.io/)

## Overview
This repository contains the code to reproduce our results by following the steps outlined below. The initial decomposition can be executed on a CPU-only machine, while the surgical rank search experiments require a single NVIDIA L4 GPU.

{Need to add comments regarding specific library dependencies here}

To be able to run the evaluation functions present in our repository it is neccessary to pull the master branch from the llm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master and run the command 'pip install -e' from inside the pulled repository.
## Supported Models 

[LLaMa - HuggingFace](https://huggingface.co/huggyllama/llama-7b)

[Mistral - HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1)

Almost all LLMs, comprising repeated modules of Attention Block + MLP Block, will be readily supported with minimal to no adjustments required for the --layer argument.
## Results

The table below shows the results of our experiments comparing Feature Space Decomposition, Weight Space Decomposition, and LLM-Pruner. The decomposition experiments apply uniform sparsity to a subset of the LLM layers :

| Decomposition  | #Params (B) | #MACS  | BoolQ | PIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | Average |
|----------------|--------------|--------|-------|-------|-----------|------------|-------|-------|---------|
| Baseline       | 6.7          | 423.93 | 75.04 | 78.67 | 76.22     | 70.00      | 72.85 | 44.88 | 69.61   |
| Feature Space (Ours)  | 5.4          | 339.99 | 74.34 | 74.86 | 66.72     | 67.40      | 66.33 | 39.42 | 64.68   |
| Weight Space   | 5.4          | 339.99 | 62.20 | 62.57 | 43.91     | 58.80      | 44.95 | 30.03 | 50.41   |
| LLM-Pruner     | 5.4          | 339.60 | 57.06 | 75.68 | 66.80     | 59.83      | 60.94 | 36.52 | 59.47   |
| Feature Space (Ours) | 3.4          | 215.61 | 62.02 | 61.37 | 34.64     | 56.43      | 40.32 | 28.75 | 47.25   |
| Weight Space   | 3.4          | 215.61 | 62.08 | 53.59 | 27.88     | 48.46      | 27.15 | 27.05 | 41.10   |
| LLM-Pruner     | 3.4          | 206.59 | 52.32 | 59.63 | 35.64     | 53.20      | 33.50 | 27.22 | 43.58   |

For detailed plots on the variation of model performance versus parameters sparsified using surgical rank search, for all common sense reasoning tasks, please refer to our [paper](https://www.arxiv.org/pdf/2405.13039).

## Steps to reproduce results 

**Step 1 :**

Run the decomposer.py script to create a model instance of choice and decompose all its layers into low rank matrices of maximum rank and create a checkpoint. (No GPU required)
#### Example
```bash
python3 decomposer.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset combination --batch_size 512 \
       --seq_len 128 \
       --log_path surgical_logs.txt \
       --algo eigen \
       --weights_name decomposed_mistral_combination.pt \
       --model mistralai/Mistral-7B-v0.1

```
**Step 2:**


To perform surgical rank search on commonsense reasoning datasets, provide the checkpoint path from the previous step as an argument to surgical.py and execute it. This script will conduct continuous evaluation for both disjoint splits (Search split and Test split). A log file will be generated to monitor the progress of the rank search and evaluation metrics. At this stage, you have the flexibility to switch the dataset to any commonsense reasoning dataset, and the performance on it will serve as a metric for the surgical rank search.
#### Example
```bash
python3 surgical.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset piqa \
       --log_path surgical_logs.txt \
       --delta 0.0 \
       --start_layer 28
       --base_model decomposed_mistral_combination.pt \
       --model mistralai/Mistral-7B-v0.1

```

#### To run rank search based on perplexity:
Run the perplexity_test.py script providing the path of the checkpoint from Step 1 as an argument. Logs will be created and evaluation on common sense reasoning tasks will be done on the entire test dataset.



