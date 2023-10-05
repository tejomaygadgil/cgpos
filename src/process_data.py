"""
Performs data processing and data cleaning.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import config
from util import is_greek, is_punctuation, read_pkl, write_pkl

# Set main dir
main_dir = Path(__file__).parents[1]


def process_raw_data():
    """
    Convert raw Perseus treebank XML data into a tabular format.
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing Perseus data:")

    # Set import and export directories
    import_dir = main_dir / config.process_import
    export_dir = main_dir / config.processed

    # Process files
    files = import_dir.iterdir()
    logger.info(f"Importing from {import_dir}")
    data = []
    for file in files:
        file_dir = import_dir / file
        logger.info(f"Processing {file}")

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
                word_attrib.update(sentence_attrib)
                word_attrib.update(work_attrib)
                data.append(word_attrib)

    # Export
    write_pkl(data, export_dir)

    logger.info(f"Success! Extracted {len(data)} words.")


def get_targets_map():
    """
    Build map to parse targets.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building targets map:")

    # Set import and export directories
    import_dir = main_dir / config.target_map_import
    export_dir = main_dir / config.target_map

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


def normalize():
    """
    Normalize Perseus data by
    - Decomposing unicode diacritics (cf. https://www.degruyter.com/document/doi/10.1515/9783110599572-009/html)
    - Stripping non-Greek characters.
    - Recomposing diacritics (for better syllablization)
    """
    logger = logging.getLogger(__name__)
    logger.info("Normalizing Perseus data:")

    # Set import and export directories
    import_dir = main_dir / config.processed
    export_dir = main_dir / config.normalized

    # Import data
    data = read_pkl(import_dir)

    # Normalize
    for word in data:
        form = word["form"]
        # Split letter from diacritics (["ί"] becomes ["ι"," ́ "])
        form = unicodedata.normalize("NFD", form)
        # Strip non-Greek chars
        form = "".join(
            [char for char in form if (is_greek(char) or is_punctuation(char))]
        )
        # Recompose (["ι"," ́ "] becomes ["ί"])
        form = unicodedata.normalize("NFC", form)

        word["norm"] = form

    # Export
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
    import_dir = main_dir / config.normalized
    targets_map_dir = main_dir / config.target_map
    cleaned_dir = main_dir / config.cleaned
    targets_dir = main_dir / config.targets

    # Import data
    data = read_pkl(import_dir)
    _, targets_str, _ = read_pkl(targets_map_dir)

    # Normalize
    cleaned = []
    targets = []
    malform = []
    for word in data:
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

    process_raw_data()
    get_targets_map()
    normalize()
    clean()
