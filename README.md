# Surgical Feature-Space Decomposition of LLMs: Why, When and How?
This repository contains the code for our paper: [Surgical Feature-Space Decomposition of LLMs: Why, When and How?](https://www.arxiv.org/pdf/2405.13039). The paper was published in Association for Computational Linguistics (ACL), [2024] by [Arnav Chavan](https://sites.google.com/view/arnavchavan/), [Nahush Lele](https://www.linkedin.com/in/nahush-lele-a06826204/), and [Deepak Gupta](https://dkgupta90.github.io/)

## Overview
This repository contains the code to reproduce our results by following the steps outlined below. The initial decomposition can be executed on a CPU-only machine, while the surgical rank search experiments require a single NVIDIA L4 GPU.
**Step 1 :**
Use run decomposer.py script to instantiate a model instance, decompose all its layers into low rank matrices of maximum rank and create a checkpoint. (No GPU required)

**Step 2:**
For surgical rank search based on commonsense reasoning datasets, use the path of the checkpoint from the previous step as an argument for surgical.py and run it. This will run the script along with continuous evaluation and checkpoints for both the disjoint splits (Search split and Test split). A log file will be created to track the progress of the rank search as well as the evaluation metrics.


**Step 3 (To run rank search based on perplexity):**
Run the perplexity_test.py script providing the path of the checkpoint from Step 1 as an arguement. Logs will be created similar to Step 2 and evaluation on common sense reasoning tasks will be done on the entire test dataset.

**Comments :**
To be able to run the evaluation functions present in our repository it is neccessary to pull the master branch from the llm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master and run the command 'pip install -e' from inside the pulled repository.

In *Step 1* different dataset names can be passed as arguements as well as an arguement as 'combination' can be passed depending on which the low rank decomposition will be done. 

Similarly in *Step 2* the rank search can be done using different datasets which can be passed as arguements to the script. 
