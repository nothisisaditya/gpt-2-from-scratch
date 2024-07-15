import os
import json
import argparse
import requests
import tiktoken
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

torch.set_float32_matmul_precision('medium')

CACHE_DIRECTORY = os.path.join(os.path.dirname(__file__), "hellaswag")

DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

tokenizer = tiktoken.get_encoding("gpt2")


def download_file(source_url: str, target_file: str, chunk_size: int = 1024):
    response = requests.get(source_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(target_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)


def download_dataset(dataset_split: str):
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    dataset_url = DATASET_URLS[dataset_split]
    dataset_file = os.path.join(CACHE_DIRECTORY, f"hellaswag_{dataset_split}.jsonl")
    if not os.path.exists(dataset_file):
        print(f"Downloading {dataset_url} to {dataset_file}...")
        download_file(dataset_url, dataset_file)


def render_example(example):
    context = example["ctx"]
    correct_ending = example["label"]
    endings = example["endings"]

    context_tokens = tokenizer.encode(context)
    ending_tokens = [tokenizer.encode(" " + ending) for ending in endings]

    max_length = max(len(context_tokens) + len(ending) for ending in ending_tokens)
    token_tensor = torch.zeros((4, max_length), dtype=torch.long)
    mask_tensor = torch.zeros((4, max_length), dtype=torch.long)

    for i, ending in enumerate(ending_tokens):
        total_length = len(context_tokens) + len(ending)
        token_tensor[i, :total_length] = torch.tensor(context_tokens + ending)
        mask_tensor[i, len(context_tokens):total_length] = 1

    return {
        "context_tokens": context_tokens,
        "ending_tokens": ending_tokens,
        "correct_ending": correct_ending,
    }, token_tensor, mask_tensor


def iterate_examples(dataset_split: str):
    download_dataset(dataset_split)
    with open(os.path.join(CACHE_DIRECTORY, f"hellaswag_{dataset_split}.jsonl"), "r") as file:
        for line in file:
            yield prepare_example(json.loads(line))


@torch.no_grad()
def evaluate(model_identifier: str):
    model = GPT2LMHeadModel.from_pretrained(model_identifier).cuda()
    model = torch.compile(model)

    total_correct = 0
    total_correct_normalized = 0
    total_examples = 0

    for example_data, tokens, mask in load_examples("val"):
        tokens = tokens.cuda()
        mask = mask.cuda()

        logits = model(tokens).logits

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_tokens = tokens[..., 1:].contiguous()
        flattened_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flattened_tokens = shifted_tokens.view(-1)
        losses = F.cross_entropy(flattened_logits, flattened_tokens, reduction='none')
        losses = losses.view(tokens.size(0), -1)

        masked_losses = losses * mask[:, 1:].contiguous()
        summed_loss = masked_losses.sum(dim=1)
        average_loss = summed_loss / mask[:, 1:].contiguous().sum(dim=1)

        predicted_ending = summed_loss.argmin().item()
        predicted_ending_normalized = average_loss.argmin().item()

        total_examples += 1
        total_correct += int(predicted_ending == example_data["correct_ending"])
        total_correct_normalized += int(
            predicted_ending_normalized == example_data["correct_ending"])

        print(
            f"{total_examples} acc_norm: {total_correct_normalized}/{total_examples}={total_correct_normalized / total_examples:.4f}")
        if total_examples < 10:
            print("---")
            print(f"Context:\n {example_data['context_tokens']}")
            print(f"Endings:")
            for i, ending in enumerate(example_data["ending_tokens"]):
                print(f"{i} (loss: {average_loss[i].item():.4f}) {ending}")
            print(
                f"predicted: {predicted_ending_normalized}, actual: {example_data['correct_ending']}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", "--model_type", type=str, default="gpt2",
                            help="Identifier for the GPT-2 model to use")
    arguments = arg_parser.parse_args()
    evaluate(arguments.model_type)
