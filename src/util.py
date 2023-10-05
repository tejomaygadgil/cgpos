"""
Contains utility functions to read data and clean Greek text.
"""
import logging
import pickle

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from greek_accentuation.characters import (
    Accent,
    Breathing,
    Diacritic,
    Length,
    Subscript,
)


# Path functions
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
