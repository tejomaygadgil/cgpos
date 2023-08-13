import os
import pickle
from pathlib import Path

from greek_accentuation.characters import (
    Accent,
    Breathing,
    Diacritic,
    Length,
    Subscript,
)


def get_abs_dir(path):
    """
    Helper function to return the absolute path from within a package (data, features, etc.)
    """
    abs_path = os.path.join(Path(__file__).resolve().parents[3], path)
    return abs_path


def import_pkl(path):
    path = get_abs_dir(path)
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def export_pkl(data, path):
    path = get_abs_dir(path)
    with open(path, "wb") as file:
        pickle.dump(data, file)


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


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
