from datasets import load_dataset
from torch.utils.data.dataset import Dataset
import random
import torch


def preprocess_function_arc(sample):
    example = {}
    label = sample["answerKey"]
    choices = sample["choices"]
    index = choices["label"].index(label)
    answer = choices["text"][index]
    example["text"] = "Question: " + sample["question"] + "\nAnswer: " + answer
    return example


def preprocess_function_openbookqa(sample):
    example = {}
    label = sample["answerKey"]
    choices = sample["choices"]
    index = choices["label"].index(label)
    answer = choices["text"][index]
    example["text"] = "Question: " + sample["question_stem"] + "\nAnswer: " + answer
    return example


def preprocess_function_gsm8k(sample):
    example = {}
    example["text"] = (
        "Question: " + sample["question"] + "\nAnswer: " + sample["answer"]
    )
    return example


def preprocess_function_hellaswag(sample):
    example = {}
    index = int(sample["label"])
    answer = sample["endings"][index]
    example["text"] = sample["ctx"] + answer
    return example


def preprocess_function_truthfulqa_mc(sample):
    example = {}
    example["text"] = sample["question"]
    return example


def preprocess_function_winogrande(sample):
    example = {}
    example["text"] = sample["sentence"].replace(
        "_", sample[f"option{sample['answer']}"]
    )
    return example


def preprocess_function_piqa(sample):
    example = {}
    example["text"] = (
        "Question: "
        + sample["goal"]
        + "\nAnswer: "
        + sample[f"sol{int(sample['label'])+1}"]
    )
    return example


def preprocess_function_boolq(sample):
    example = {}
    example[
        "text"
    ] = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer: {sample['answer']}"
    return example


def preprocess_function_mnli(sample):
    example = {}
    example[
        "text"
    ] = f"mnli premise: {sample['premise']} hypothesis: {sample['hypothesis']} target:"
    return example


def preprocess_function_sst2(sample):
    example = {}
    example["text"] = f"sst2 sentence: {sample['sentence']} label:"
    return example


def preprocess_function_stsb(sample):
    example = {}
    example[
        "text"
    ] = f"stsb sentence1: {sample['sentence1']} sentence2: {sample['sentence2']} label:"
    return example


def get_dataset(dataset_name):
    if dataset_name == "mnli":
        dataset = load_dataset("multi_nli", split="train")
        dataset_eval = load_dataset("multi_nli", split="validation_matched")
        preprocess_function = preprocess_function_mnli
        ind = range(100000)
        dataset = dataset.select(ind)
        label_map = ["entailment", "neutral", "contradiction"]
        true_labels = [label_map[example["label"]] for example in dataset_eval]

    elif dataset_name == "boolq":
        dataset = load_dataset("boolq", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_boolq

    elif dataset_name == "sst2":
        dataset = load_dataset("sst2", split="train")
        dataset_eval = load_dataset("sst2", split="validation")
        preprocess_function = preprocess_function_sst2
        true_labels = [
            "positive" if example["label"] == 1 else "negative"
            for example in dataset_eval
        ]

    elif dataset_name == "stsb":
        dataset = load_dataset("glue", "stsb", split="train")
        preprocess_function = preprocess_function_stsb

    elif dataset_name == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_hellaswag

    elif dataset_name == "truthfulqa_mc":
        dataset = load_dataset("EleutherAI/truthful_qa_mc", split="validation")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_truthfulqa_mc

    elif dataset_name == "arc_challenge":
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_arc

    elif dataset_name == "arc_easy":
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_arc

    elif dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_gsm8k

    elif dataset_name == "winogrande":
        dataset = load_dataset("winogrande", "winogrande_xl", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_winogrande

    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_piqa

    elif dataset_name == "openbookqa":
        dataset = dataset = load_dataset("openbookqa", "main", split="train")
        dataset_eval = None
        true_labels = None
        preprocess_function = preprocess_function_openbookqa

    dataset = dataset.map(preprocess_function)
    if dataset_eval:
        dataset_eval = dataset_eval.map(preprocess_function)  # .select(ind)
        return dataset, dataset_eval, true_labels
    return dataset, None, None


######### Generalised Data
def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )