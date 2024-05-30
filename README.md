# Surgical Feature-Space Decomposition of LLMs: Why, When and How?
This repository contains the code for our paper: [Surgical Feature-Space Decomposition of LLMs: Why, When and How?](https://www.arxiv.org/pdf/2405.13039). The paper was published in Association for Computational Linguistics (ACL), [2024] by [Arnav Chavan](https://sites.google.com/view/arnavchavan/), [Nahush Lele](https://www.linkedin.com/in/nahush-lele-a06826204/), and [Deepak Gupta](https://dkgupta90.github.io/)

## Overview
This repository contains the code to reproduce our results by following the steps outlined below. The initial decomposition can be executed on a CPU-only machine, while the surgical rank search experiments require a single NVIDIA L4 GPU.

{Need to add comments regarding specific library dependencies here}

To be able to run the evaluation functions present in our repository it is neccessary to pull the master branch from the llm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master and run the command 'pip install -e' from inside the pulled repository.
## Supported Models 

[LLaMa - HuggingFace](https://huggingface.co/huggyllama/llama-7b)

[Mistral - HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1)

Almost all LLMs, comprising of repeated modules of Attention Block + MLP Block, will be readily supported with minimal to no adjustments required for the --layer argument.
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

#### Results for Perplexity Based Surgical Rank Search
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Datasets</th>
      <th colspan="9">Budget</th>
    </tr>
    <tr>
      <th>100%</th>
      <th>94%</th>
      <th>87%</th>
      <th>83%</th>
      <th>79%</th>
      <th>75%</th>
      <th>70%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">LLaMa-7b</td>
      <td>PIQA</td>
      <td>78.67</td>
      <td>76.82</td>
      <td>76.39</td>
      <td>75.13</td>
      <td>73.55</td>
      <td>71.71</td>
      <td>71.27</td>
    </tr>
    <tr>
      <td>BoolQ</td>
      <td>75.04</td>
      <td>73.61</td>
      <td>72.26</td>
      <td>73.21</td>
      <td>71.07</td>
      <td>66.02</td>
      <td>64.92</td>
    </tr>
    <tr>
      <td>ARC-C</td>
      <td>44.88</td>
      <td>42.92</td>
      <td>42.24</td>
      <td>41.38</td>
      <td>40.01</td>
      <td>36.86</td>
      <td>35.07</td>
    </tr>
    <tr>
      <td>ARC-E</td>
      <td>72.85</td>
      <td>71.46</td>
      <td>68.56</td>
      <td>66.50</td>
      <td>64.18</td>
      <td>60.48</td>
      <td>55.26</td>
    </tr>
    <tr>
      <td>Winogrande</td>
      <td>70.00</td>
      <td>69.29</td>
      <td>69.46</td>
      <td>69.37</td>
      <td>67.56</td>
      <td>62.67</td>
      <td>56.35</td>
    </tr>
    <tr>
      <td>Hellaswag</td>
      <td>76.22</td>
      <td>74.15</td>
      <td>71.65</td>
      <td>69.09</td>
      <td>65.67</td>
      <td>60.29</td>
      <td>52.62</td>
    </tr>
    <tr>
      <td>Average</td>
      <td>69.61</td>
      <td>68.04</td>
      <td>66.79</td>
      <td>65.78</td>
      <td>63.68</td>
      <td>59.67</td>
      <td>55.92</td>
    </tr>
  </tbody>
</table>


<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Datasets</th>
      <th colspan="9">Budget</th>
    </tr>
    <tr>
      <th>100%</th>
      <th>94%</th>
      <th>87%</th>
      <th>83%</th>
      <th>79%</th>
      <th>75%</th>
      <th>70%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">Mistral-7b</td>
      <td>PIQA</td>
      <td>80.52</td>
      <td>80.01</td>
      <td>78.73</td>
      <td>78.84</td>
      <td>76.99</td>
      <td>75.68</td>
      <td>75.84</td>
    </tr>
    <tr>
      <td>BoolQ</td>
      <td>83.58</td>
      <td>81.38</td>
      <td>81.56</td>
      <td>81.34</td>
      <td>78.96</td>
      <td>76.54</td>
      <td>73.33</td>
    </tr>
    <tr>
      <td>ARC-C</td>
      <td>54.01</td>
      <td>52.90</td>
      <td>49.74</td>
      <td>47.61</td>
      <td>43.51</td>
      <td>38.22</td>
      <td>37.20</td>
    </tr>
    <tr>
      <td>ARC-E</td>
      <td>79.54</td>
      <td>79.38</td>
      <td>78.37</td>
      <td>77.65</td>
      <td>74.83</td>
      <td>71.93</td>
      <td>70.41</td>
    </tr>
    <tr>
      <td>Winogrande</td>
      <td>74.03</td>
      <td>74.51</td>
      <td>73.56</td>
      <td>72.06</td>
      <td>70.80</td>
      <td>65.43</td>
      <td>64.48</td>
    </tr>
    <tr>
      <td>Hellaswag</td>
      <td>81.05</td>
      <td>79.91</td>
      <td>77.65</td>
      <td>75.77</td>
      <td>71.80</td>
      <td>66.13</td>
      <td>60.37</td>
    </tr>
    <tr>
      <td>Average</td>
      <td>75.46</td>
      <td>74.69</td>
      <td>73.27</td>
      <td>72.21</td>
      <td>69.48</td>
      <td>65.56</td>
      <td>63.60</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th rowspan="3">Model</th>
      <th rowspan="3">Dataset</th>
      <th colspan="8">Layers Pruned</th>
    </tr>
    <tr>
      <th colspan="2">0</th>
      <th colspan="2">35</th>
      <th colspan="2">70</th>
      <th colspan="2">140</th>
    </tr>
    <tr>
      <th>Accuracy</th>
      <th>Budget</th>
      <th>Accuracy</th>
      <th>Budget</th>
      <th>Accuracy</th>
      <th>Budget</th>
      <th>Accuracy</th>
      <th>Budget</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">LLaMA-7B</td>
      <td>PIQA</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>BoolQ</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ARC-C</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ARC-E</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Winogrande</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hellaswag</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="6">Mistral-7B</td>
      <td>PIQA</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>BoolQ</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ARC-C</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ARC-E</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Winogrande</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Hellaswag</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


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




