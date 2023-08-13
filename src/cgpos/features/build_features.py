# -*- coding: utf-8 -*-
import logging
import xml.etree.ElementTree as ET

import hydra
from greek_accentuation.syllabify import syllabify
from omegaconf import DictConfig

from cgpos.utils.util import export_pkl, get_abs_dir, import_pkl


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def make_postag_map(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Making postag map:")

    # Set import and export directories
    file_dir = get_abs_dir(config.perseus.tagset)
    export_dir = config.perseus.postag_map

    # Load XML
    tree = ET.parse(file_dir)
    root = tree.getroot()

    # Build map
    data = {}
    for i, element in enumerate(next(root.iter("attributes"))):
        category = element.tag
        data[(i, "-")] = (category, "N/A")
        for value in element.find("values"):
            postag = value.find("postag").text
            description = value.find("long").text
            data[(i, postag)] = (category, description)

    logger.info(f"Success! Built map with {len(data)} part-of-speech categories.")

    # Export as pickle
    logger.info(f"Exporting to {export_dir}")
    export_pkl(data, export_dir)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def featurize_perseus(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Building Perseus features:")

    # Set import and export directories
    import_dir = config.perseus.normalized
    postag_dir = config.perseus.postag_map
    export_dir = config.perseus.featurized

    # Import data
    data = import_pkl(import_dir)
    postag_map = import_pkl(postag_dir)

    bad_pos = []
    for word_dict in data:
        # Parse syllables
        word_dict["syllables"] = syllabify(word_dict["normalized"])
        # Parse part-of-speech
        if "postag" in word_dict:
            for i, tag in enumerate(word_dict["postag"]):
                key = (i, tag)
                try:
                    pos_category, pos_value = postag_map[key]
                    word_dict[pos_category] = pos_value
                except KeyError:
                    match key:
                        case (5, "d"):  # Treat depondent verbs as medio-passive
                            word_dict["voice"] = "medio-passive"
                        case (0, "x"):  # Accept irregular pos tag
                            word_dict["pos"] = "irregular"
                        case _:
                            bad_pos.append(word_dict)

    logging.info("Success! Syllablized normalized form and parsed part-of-speech tags.")

    # Export
    logging.info(f"Exporting to {export_dir}")
    export_pkl(data, export_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    make_postag_map()
    featurize_perseus()
