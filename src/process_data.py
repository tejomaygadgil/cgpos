"""
Performs data processing and data cleaning.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from beta_code import beta_code_to_greek
from greek_accentuation.syllabify import syllabify
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from util import is_greek, is_punctuation, read_pkl, write_pkl


def read_raw(read_dir, write_dir, postag):
    """
    Convert raw XML data into a tabular format.
    """
    logger = logging.getLogger(__name__)
    files = sorted(list(Path(read_dir).iterdir()))
    logger.info(f"Reading {len(files)} files from {read_dir}")
    data = []
    for file_dir in tqdm(files):
        with logging_redirect_tqdm():
            logger.info(f"Processing {file_dir}")
        tree = ET.parse(file_dir)
        root = tree.getroot()
        for sentence in root.iter("sentence"):
            for node in sentence.iter():
                word_data = {}
                match node.tag:
                    case "word":
                        word_data["form"] = node.attrib.get("form")
                        if postag:
                            word_data["postag"] = node.attrib.get("postag")
                    case "punct":
                        word_data["form"] = node.attrib.get("mark")
                data.append(word_data)

    # Export
    write_pkl(data, write_dir)
    logger.info(f"Success! Extracted {len(data)} words: {data[:10]}")


def read_targets_map():
    """
    Build map to parse targets.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building targets map:")

    # Set import and export directories
    import_dir = cfg.ft_raw_targets_map
    export_dir = cfg.ft_targets_map

    # Build map
    tree = ET.parse(import_dir)
    root = tree.getroot()
    data = ([], [], [])  # POS category, class, int
    for element in next(root.iter("attributes")):
        pos_class = element.tag
        data[0].append(pos_class)
        data[1].append([])
        data[2].append([])
        data[1][-1].append("-")
        data[2][-1].append("N/A")
        for value in element.find("values"):
            short = value.find("postag").text
            long = value.find("long").text
            if long not in ["none of the above", "I do not know"]:
                data[1][-1].append(short)
                data[2][-1].append(long)

    # Export
    write_pkl(data, export_dir)
    logger.info(f"Success! Built targets map for {len(data[0])} targets: {data[0]}")


def beta2uni():
    logger = logging.getLogger(__name__)
    logger.info("Converting Beta Code to Greek Unicode:")
    data = read_pkl(cfg.pt_processed)
    for word_data in tqdm(data):
        word_data["form"] = beta_code_to_greek(word_data["form"])

    write_pkl(data, cfg.pt_processed)


def normalize(read_dir, write_dir):
    """
    - Decompose unicode diacritics (cf. https://www.degruyter.com/document/doi/10.1515/9783110599572-009/html)
    - Strip non-Greek characters.
    - Recompose diacritics (for better syllablization)
    """
    logger = logging.getLogger(__name__)
    logger.info("Normalizing data:")
    data = read_pkl(read_dir)
    for word_data in tqdm(data):
        try:  # Decompose and recompose, dropping non-Greek characters
            word = word_data["form"]
            norm = unicodedata.normalize("NFD", word)
            norm = "".join([ch for ch in norm if (is_greek(ch) or is_punctuation(ch))])
            norm = unicodedata.normalize("NFC", norm)
            word_data["form"] = norm
        except KeyError:  # Skip over words missing keys (pre-trained data)
            pass

    # Export
    write_pkl(data, write_dir)
    logger.info(
        "Success! Performed unicode normalization and stripped non-Greek characters."
    )


def clean():
    """
    Clean normalized data for training:

    - Drop malformed words
    - Build targets for training.

    Export: [form_1, ...], [[target_1_1, target_1_2, ...], ...]
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data for training:")

    # Set import and export directories
    import_dir = cfg.ft_normalized
    targets_map_dir = cfg.ft_targets_map
    cleaned_dir = cfg.ft_text
    targets_dir = cfg.ft_targets

    # Import data
    data = read_pkl(import_dir)
    _, targets_str, _ = read_pkl(targets_map_dir)

    # Normalize
    cleaned = []
    targets = []
    malform = 0
    for word in tqdm(data):
        try:
            assert "postag" in word and "form" in word
            assert word["form"]
            assert word["postag"]
            assert len(word["postag"]) == 9
            assert word["postag"] != "undefined"
            # Build features
            form = word["form"]
            # Build target
            target = []
            postag = word["postag"]
            for i, short in enumerate(postag):
                match (i, short):
                    case ("5", "d"):  # Treat depondent verbs as medio-passive
                        i, short = "5", "e"
                value = targets_str[i].index(short)
                target.append(value)
            # Append
            cleaned.append(form)
            targets.append(target)
        except (AssertionError, ValueError, TypeError):
            malform += 1

    length_match = len(cleaned) == len(targets)
    assert (
        length_match
    ), f"Syllables and target lengths do not match: {len(cleaned[0])}, {len(targets[1])}"

    # Export
    write_pkl(cleaned, cleaned_dir)
    write_pkl(targets, targets_dir)

    logger.info(
        f"Success! Exporting {len(cleaned)} words (dropped {malform} malformed words)."
    )


def syllablize(read_dir, write_dir):
    logger = logging.getLogger(__name__)
    text = read_pkl(read_dir)
    logger.info(f"Syllabifying {len(text)} words: {text[:10]}")
    tokens = []
    for word in tqdm(text):
        syllables = syllabify(word) + [" "]
        tokens.extend(syllables)

    # Export
    write_pkl(tokens, write_dir)
    logger.info(f"Success! Generated {len(tokens)} syllables: {tokens[:10]}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    match sys.argv[1]:
        case "pt":  # Pre-training
            read_raw(cfg.pt_raw, cfg.pt_processed, postag=False)
            # normalize(cfg.pt_processed, cfg.pt_text)
            # syllablize(cfg.pt_text, cfg.pt_syl)

        case "ft":  # Fine-tuning
            read_raw(cfg.ft_raw, cfg.ft_processed, postag=True)
            read_targets_map()  # -> cfg.ft_targets
            normalize(cfg.ft_processed, cfg.ft_normalized)
            clean()  # cfg.ft_normalized, cfg.ft_targets_map -> cfg.ft_text, cfg.ft_targets
            syllablize(cfg.ft_text, cfg.ft_syl)

        case _:
            "Please select either 'pt' or 'ft' as options."
