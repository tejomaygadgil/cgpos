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
    abs_dir = get_abs_dir()
    import_dir = os.path.join(abs_dir, config.data_dir.perseus)
    export_dir = os.path.join(abs_dir, config.data_dir.processed, "perseus.pkl")

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

        # Iterate through sentences
        for sentence in root.iter("sentence"):
            sentence_attrib = sentence.attrib
            sentence_attrib["sentence_id"] = sentence_attrib.pop("id")  # Rename
            for word in sentence:
                word_attrib = word.attrib
                word_attrib.update(sentence_attrib)
                data.append(word_attrib)

    logging.info(f"Success! Extracted {len(data)} words from {len(files)} files.")

    # Export as pickle
    logging.info(f"Exporting to {export_dir}")
    with open(export_dir, "wb") as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    read_perseus()
