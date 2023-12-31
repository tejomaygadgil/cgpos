"""
Performs data processing and data cleaning.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os
import unicodedata
import xml.etree.ElementTree as ET

import hydra
from greek_accentuation.syllabify import syllabify
from omegaconf import DictConfig

from cgpos.util.greek import is_greek, is_punctuation
from cgpos.util.path import export_pkl, get_abs_dir, import_pkl


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def process_raw_data(config: DictConfig):
    """
    Convert raw Perseus treebank XML data into a tabular format.
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing Perseus data:")

    # Set import and export directories
    import_dir = get_abs_dir(config.perseus.raw_dir)
    export_dir = config.perseus.raw

    # Get files
    files = os.listdir(import_dir)
    logger.info(f"Importing {len(files)} files from {import_dir}")

    # Collect words from files
    data = []
    for file in files:
        file_dir = os.path.join(import_dir, file)
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
    export_pkl(data, export_dir)

    logger.info(f"Success! Extracted {len(data)} words from {len(files)} files.")


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def get_targets_map(config: DictConfig):
    """
    Build map to parse targets.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building targets map:")

    # Set import and export directories
    file_dir = get_abs_dir(config.perseus.tagset)
    export_dir = config.reference.targets_map

    # Load XML
    tree = ET.parse(file_dir)
    root = tree.getroot()

    # Build map
    data = ([], [], [])
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
    export_pkl(data, export_dir)

    logger.info(f"Success! Built targets map for {len(data[0])} targets: {data[0]}")


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def normalize(config: DictConfig):
    """
    Normalize Perseus data by
    - Decomposing unicode diacritics (cf. https://www.degruyter.com/document/doi/10.1515/9783110599572-009/html)
    - Stripping non-Greek characters.
    - Recomposing diacritics (for better syllablization)
    """
    logger = logging.getLogger(__name__)
    logger.info("Normalizing Perseus data:")

    # Set import and export directories
    import_dir = config.perseus.raw
    export_dir = config.perseus.normalized

    # Import data
    data = import_pkl(import_dir)

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
    export_pkl(data, export_dir)

    logger.info(
        "Success! Performed unicode normalization and stripped non-Greek characters."
    )


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def clean(config: DictConfig):
    """
    Clean normalized data for training:

    - Drop malformed words
    - Build targets for training.

    Export: [[sentence_id_1], ...], [[form_1], ...], [[target_1], ...]]
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data for training:")

    # Set import directories
    import_dir = config.perseus.normalized
    targets_map_dir = config.reference.targets_map
    # Set export directories
    export_dir = config.data.cleaned

    # Import data
    data = import_pkl(import_dir)
    _, targets_short, _ = import_pkl(targets_map_dir)

    # Normalize
    cleaned = [[], []]
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
            syllables = syllabify(norm)
            # Build target
            target = []
            postag = word["postag"]
            for i, short in enumerate(postag):
                match (i, short):
                    case ("5", "d"):  # Treat depondent verbs as medio-passive
                        i, short = "5", "e"
                value = targets_short[i].index(short)
                target.append(value)
            # Append
            cleaned[0].append(syllables)
            cleaned[1].append(target)
        except (AssertionError, ValueError):
            malform.append(word)

    length_match = len(cleaned[0]) == len(cleaned[1])
    assert length_match, "Syllables and target lengths do not match."

    # Export
    export_pkl(cleaned, export_dir)

    logger.info(
        f"Success! Exporting {len(cleaned[0])} words (dropped {len(malform)} malformed words)."
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    process_raw_data()
    get_targets_map()
    normalize()
    clean()
