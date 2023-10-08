"""
Contains utility functions to read data and clean Greek text.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import pickle
import sys

import torch
from greek_accentuation.characters import (
    Accent,
    Breathing,
    Diacritic,
    Length,
    Subscript,
)


# TRAIN
# Tokenizing
def encode(stoi, text):
    return [stoi[c] for c in text]


def decode(itos, tokens):
    return "".join([itos[i] for i in tokens])


# Data loading
def get_batch(data, block_size, batch_size, device, y=None):
    # Generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    if y is not None:
        y = torch.stack([y[i + 1 : i + block_size + 1] for i in ix])
    else:
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def generate(length, block_size, itos, model, device):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    text = decode(itos, model.generate(context, length, block_size)[0].tolist())
    model.train()
    return text


# FUN
def display_bar(data, line_len=50):
    out = ""
    for i, v in enumerate(data):
        if i % 50 == 0:
            out += "\n"
        match v:
            case 1:
                out += "\033[42m \033[0m"
            case 0:
                out += "\033[41m \033[0m"
    out += "\n"
    sys.stdout.write(out)
    sys.stdout.flush()


# PATHING
def read_pkl(path, verbose=True):
    logger = logging.getLogger(__name__)
    with open(path, "rb") as file:
        data = pickle.load(file)
    if verbose:
        logger.info(f"Imported {path}")
    return data


def write_pkl(data, path, verbose=True):
    logger = logging.getLogger(__name__)
    with open(path, "wb") as file:
        pickle.dump(data, file)
    if verbose:
        logger.info(f"Exported {path}")


# GREEK
# Some sets of Unicode Greek characters. Useful for normalization
GREEK_LOWER = set(range(0x03B1, 0x03CA))
GREEK_UPPER = set(range(0x0391, 0x03AA))
GREEK_CHARACTERS = set.union(GREEK_LOWER, GREEK_UPPER)
GREEK_DIACRITICS = {  # Data from `greek_accentuation` library
    ord(mark.value)
    for mark_set in [
        Breathing,
        Accent,
        Diacritic,
        Subscript,
        Length,
    ]
    for mark in mark_set
    if (type(mark.value) is not int)  # Breathing contains -1 value for some reason
}
GREEK_MARKS = set.union(GREEK_CHARACTERS, GREEK_DIACRITICS)
GREEK_PUNCTUATION = {  # cf. https://www.degruyter.com/document/doi/10.1515/9783110599572-009/html p. 153
    0x002C,  # Comma
    0x002E,  # Full stop
    0x003B,  # Semicolon (question)
    0x00B7,  # Middle dot (semicolon)
    # 0x02B9, # Modifier letter prime (?) -- Not in use
    0x2019,  # Right single quotation mark (elision)
}


def is_greek(char):
    """
    Determine if char is Greek.
    """
    return ord(char) in GREEK_MARKS


def is_punctuation(char):
    """
    Determine if char is Greek punctuation.
    """
    return ord(char) in GREEK_PUNCTUATION
