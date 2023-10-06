"""
Performs data processing and data cleaning.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import config as cfg
from util import is_greek, is_punctuation, read_pkl, write_pkl


def read_raw(import_dir, export_dir):
    """
    Convert raw XML data into a tabular format.
    """
    logger = logging.getLogger(__name__)
    files = sorted(list(Path(import_dir).iterdir()))
    logger.info(f"Reading {len(files)} files from {import_dir}")
    data = []
    for file_dir in tqdm(files):
        with logging_redirect_tqdm():
            logger.info(f"Processing {file_dir}")

        # Load XML
        tree = ET.parse(file_dir)
        root = tree.getroot()

        # Get title and author
        work_attrib = {}
        for field in ["author", "title"]:
            element = list(root.iter(field))
            if len(element):  # Some files are missing author info
                work_attrib[field] = element[0].text

        # Get POS tags
        for sentence in root.iter("sentence"):
            sentence_attrib = sentence.attrib.copy()
            sentence_attrib["sentence_id"] = sentence_attrib.pop("id")
            for word in sentence.iter("word"):
                word_attrib = word.attrib.copy()
                lemma = word.find("lemma")  # Pre-training data has lemma
                if lemma:
                    lemma_attrib = lemma.attrib.copy()
                    lemma_attrib.update(work_attrib)
                    lemma_attrib.update(sentence_attrib)
                    lemma_attrib.update(word_attrib)
                    data.append(lemma_attrib)
                else:
                    word_attrib.update(work_attrib)
                    word_attrib.update(sentence_attrib)
                    data.append(word_attrib)

    # Export
    write_pkl(data, export_dir)
    logger.info(f"Success! Extracted {len(data)} words.")


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


def normalize(import_dir, export_dir, word_key):
    """
    Normalize Perseus data by
    - Decomposing unicode diacritics (cf. https://www.degruyter.com/document/doi/10.1515/9783110599572-009/html)
    - Stripping non-Greek characters.
    - Recomposing diacritics (for better syllablization)
    """
    logger = logging.getLogger(__name__)
    logger.info("Normalizing data:")
    data = read_pkl(import_dir)
    if word_key == "entry":
        normed = []
    for word_data in tqdm(data):
        try:  # Decompose and recompose, dropping non-Greek characters
            word = word_data[word_key]
            norm = unicodedata.normalize("NFD", word)
            norm = "".join([ch for ch in norm if (is_greek(ch) or is_punctuation(ch))])
            norm = unicodedata.normalize("NFC", norm)
            if word_key == "entry":
                normed.append(norm)
            else:
                word_data["norm"] = norm
        except KeyError:  # Skip over words missing keys (pre-trained data)
            pass

    # Export
    if word_key == "entry":
        write_pkl(normed, export_dir)
    else:
        write_pkl(data, export_dir)
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
    cleaned_dir = cfg.ft_cleaned
    targets_dir = cfg.ft_targets

    # Import data
    data = read_pkl(import_dir)
    _, targets_str, _ = read_pkl(targets_map_dir)

    # Normalize
    cleaned = []
    targets = []
    malform = []
    for word in tqdm(data):
        try:
            # Check form
            assert "postag" in word and "norm" in word
            assert len(word["postag"]) == 9
            assert word["postag"] != "undefined"
            assert word["norm"]
            # Build features
            norm = word["norm"]
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
            cleaned.append(norm)
            targets.append(target)
        except (AssertionError, ValueError):
            malform.append(word["form"])

    length_match = len(cleaned) == len(targets)
    assert (
        length_match
    ), f"Syllables and target lengths do not match: {len(cleaned[0])}, {len(targets[1])}"

    # Export
    write_pkl(cleaned, cleaned_dir)
    write_pkl(targets, targets_dir)

    logger.info(
        f"Success! Exporting {len(cleaned)} words (dropped {len(malform)} malformed words)."
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    match sys.argv[1]:
        case "pt":  # Pre-training (read and normalize)
            # read_raw(cfg.pt_raw, cfg.pt_processed)
            normalize(cfg.pt_processed, cfg.pt_text, word_key="entry")

        case "ft":  # Fine-tuning (read data, read map, normalize, and clean)
            read_raw(cfg.ft_raw, cfg.ft_processed)
            read_targets_map()
            normalize(cfg.ft_processed, cfg.ft_normalized, word_key="form")
            clean()

        case _:
            "Please select either 'pt' or 'ft' as options."
