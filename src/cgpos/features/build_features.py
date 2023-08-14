# -*- coding: utf-8 -*-
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict

import hydra
from greek_accentuation.syllabify import syllabify
from omegaconf import DictConfig

from cgpos.utils.util import export_pkl, get_abs_dir, import_pkl


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def get_postag_map(config: DictConfig):
    """
    Parse part-of-speech map to extract features from `postag` column.
    """
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

    # Add irregularities
    data[(5, "d")] = ("voice", "medio-passive")  # Treat depondent as medio-passive
    data[(0, "x")] = ("pos", "irregular")  # Accept irregular pos tag

    logger.info("Success! Built map to parse part-of-speech categories.")

    # Export as pickle
    export_pkl(data, export_dir)

    logger.info("Mapping part-of-speech categories to values:")
    # Set export directory
    export_dir = config.perseus.category_map

    # Create category-value map
    category_map = defaultdict(list)
    for category, value in data.values():
        category_map[category].append(value)

    logger.info(
        f"Success! Built map with {len(category_map)} part-of-speech categories."
    )

    # Export as pickle
    export_pkl(category_map, export_dir)

    logger.info("Mapping cat2int (and vice-versa):")
    # Set export directory
    cat2int_export_dir = config.perseus.cat2int
    int2cat_export_dir = config.perseus.int2cat

    # Create cat2int and int2cat map
    cat2int = {}
    int2cat = {}
    for category, values in category_map.items():
        for i, value in enumerate(values):
            cat2int[(category, value)] = i
            int2cat[(category, i)] = value

    logger.info("Success! Built cat2int and int2cat maps.")

    # Export as pickle
    export_pkl(cat2int, cat2int_export_dir)
    export_pkl(int2cat, int2cat_export_dir)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def featurize(config: DictConfig):
    """
    Featurize Perseus data by
    - Getting syllables from normalized form (via greek_accentuation lib)
    - Parsing part-of-speech tag data (via map generated from parse_postag)
    """
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
        word_dict["syllables"] = syllabify(word_dict["norm"])
        # Parse part-of-speech
        if "postag" in word_dict:
            for i, tag in enumerate(word_dict["postag"]):
                key = (i, tag)
                try:
                    pos_category, pos_value = postag_map[key]
                    word_dict[pos_category] = pos_value
                except KeyError:
                    bad_pos.append(word_dict)

    logger.info("Success! Syllablized normalized form and parsed part-of-speech tags.")

    # Export
    export_pkl(data, export_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logging.info("2. BUILDING FEATURES")

    get_postag_map()
    featurize()
