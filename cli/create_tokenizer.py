
import argparse
import logging

import datasets
from datasets import load_dataset

import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer")

    parser.add_argument("--vocab_size", type=int, required=True, help="Size of the vocabulary")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory which will be used to save tokenizer.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger.info(f"Starting tokenizer training with args {args}")

    # Use Datsets to load wikitext corpus. This is a very relatively dataset, less than 1Gb.
    # wikitext-103-v1 is a particular version of this dataset, you can look for other versions here: https://huggingface.co/datasets/wikitext
    logger.info(f"Loading wikitext dataset")
    raw_datasets = load_dataset("wikitext", "wikitext-103-v1")

    logger.info(f"Building tokenizer (might take a couple of minutes)")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Task 3.1: train a BPE tokenizer
    # You only need ["UNK"] and ["PAD"] special tokens.
    # DO NOT just copy everything from the tutorial.
    # Use vocab_size=args.vocab_size.
    # When you run this sript use 8192. The model should converge faster with a smaller vocab size.
    # Tokenizer training tutorial: https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
    # API reference for tokenizer.trainer class: 

    # YOUR CODE STARTS HERE (our implementation is about 6 lines)

    tokenizer = Tokenizer(BPE(special_tokens=["[UNK]", "[PAD]"]), max_model_length=args.vocab_size)
    tokenizer_trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"])
    tokenizer.pre_tokenizer = Whitespace()

    iterator = (item["text"] for item in raw_datasets["train"])
    tokenizer.train_from_iterator(iterator, tokenizer_trainer, len(raw_datasets["train"]))
    # YOUR CODE ENDS HERE

    # wrap the tokenizer to make it usable in HuggingFace Transformers
    tokenizer = transformers.PreTrainedTokenizerFast(args.vocab_size, tokenizer_object=tokenizer)
    logger.info(f"Saving tokenizer to {args.save_dir}")
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__" :
    main()
