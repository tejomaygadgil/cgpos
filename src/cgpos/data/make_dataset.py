# -*- coding: utf-8 -*-
import logging
import os
import unicodedata
import xml.etree.ElementTree as ET

import hydra
from omegaconf import DictConfig

from cgpos.utils.util import (
    export_pkl,
    get_abs_dir,
    import_pkl,
    is_greek,
    is_punctuation,
)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def read_perseus(config: DictConfig):
    """
    Converts raw Perseus treebank XML data into a tabular format.
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing Perseus data:")

    # Set import and export directories
    import_dir = get_abs_dir(config.perseus.raw_dir)
    export_dir = get_abs_dir(config.perseus.processed)

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
            sentence_attrib["sentence_id"] = sentence_attrib.pop("id")  # Rename
            for word in sentence.iter("word"):
                word_attrib = word.attrib.copy()
                word_attrib.update(sentence_attrib)
                word_attrib.update(work_attrib)
                data.append(word_attrib)

    logging.info(f"Success! Extracted {len(data)} words from {len(files)} files.")

    # Export as pickle
    logging.info(f"Exporting to {export_dir}")
    export_pkl(data, export_dir)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def normalize(config: DictConfig):
    logging.info("Normalizing Perseus data:")

    # Set import and export directories
    import_dir = config.perseus.processed
    export_dir = config.perseus.normalized

    # Import data
    data = import_pkl(import_dir)

    # Normalize
    for word_dict in data:
        word = word_dict["form"]
        # Split letter from diacritics: ["ί"] becomes ["ι"," ́ "]
        word = unicodedata.normalize("NFD", word)
        # Strip non-Greek chars
        word = "".join(
            [char for char in word if (is_greek(char) or is_punctuation(char))]
        )
        word_dict["normalized"] = word

    logging.info("Success! Decomposed diacritics and stripped non-Greek characters.")

    # Export
    logging.info(f"Exporting to {export_dir}")
    export_pkl(data, config.perseus.normalized)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    read_perseus()
    normalize()
