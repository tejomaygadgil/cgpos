# -*- coding: utf-8 -*-
import logging
import os
import pickle
import xml.etree.ElementTree as ET

import hydra
from omegaconf import DictConfig

from cgpos.utils.util import get_abs_dir


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
            for word in sentence:
                word_attrib = word.attrib.copy()
                word_attrib.update(sentence_attrib)
                word_attrib.update(work_attrib)
                data.append(word_attrib)

    logging.info(f"Success! Extracted {len(data)} words from {len(files)} files.")

    # Export as pickle
    logging.info(f"Exporting to {export_dir}")
    with open(export_dir, "wb") as file:
        pickle.dump(data, file)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def load_perseus(config: DictConfig):
    """
    Loads processed perseus.pkl.
    """
    perseus_path = get_abs_dir(config.perseus.processed)

    with open(perseus_path, "rb") as file:
        data = pickle.load(file)

    return data


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    read_perseus()
