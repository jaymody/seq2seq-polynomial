import re
import random

from tqdm import tqdm
from torchtext.data import Dataset, Example, Field, BucketIterator

from utils import get_device

device = get_device()


def tokenize(text):
    return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", text)


def create_examples(factors, expansions):
    src = Field(
        tokenize=tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
        batch_first=True,
    )

    trg = Field(
        tokenize=tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
        batch_first=True,
    )

    examples = []
    for factor, expansion in tqdm(zip(factors, expansions), desc="creating examples"):
        examples.append(
            Example.fromlist([factor, expansion], fields=[("src", src), ("trg", trg)])
        )

    return examples, src, trg


def train_test_split(examples, train_test_split_ratio=0.9):
    random.shuffle(examples)
    split = int(train_test_split_ratio * len(examples))
    train_examples, test_examples = examples[:split], examples[split:]
    return train_examples, test_examples


def create_iterators(train_examples, test_examples, src, trg, batch_size=128):
    train_data = Dataset(train_examples, fields={"src": src, "trg": trg})
    test_data = Dataset(test_examples, fields={"src": src, "trg": trg})

    src.build_vocab(train_data, test_data, min_freq=1)
    trg.build_vocab(train_data, test_data, min_freq=1)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data, test_data),
        batch_size=batch_size,
        sort=False,
        device=device,
    )
    return train_iterator, valid_iterator, test_iterator
