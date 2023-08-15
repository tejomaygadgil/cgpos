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
    export_dir = config.data.target_map

    # Load XML
    tree = ET.parse(file_dir)
    root = tree.getroot()

    # Build map
    data = defaultdict(list)
    for element in next(root.iter("attributes")):
        pos_class = element.tag
        data[pos_class].append(("-", "N/A"))
        for _i, value in enumerate(element.find("values"), start=1):
            short = value.find("postag").text
            long = value.find("long").text
            data[pos_class].append((short, long))

    # Accept irregular pos tag
    data["pos"].append(("x", "irregular"))

    logger.info("Success! Built map to parse part-of-speech categories.")

    # Export as pickle
    export_pkl(data, export_dir)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def tokenize(config: DictConfig):
    """
    Tokenize features and target.
    """
    logger = logging.getLogger(__name__)
    logger.info("Tokenizing Perseus features and targets:")

    # Import directories
    import_dir = config.perseus.normalized
    target_map_dir = config.data.target_map
    # Export directories
    feature_map_dir = config.data.feature_map
    features_dir = config.data.features
    targets_dir = config.data.targets

    # Import data
    data = import_pkl(import_dir)
    target_map = import_pkl(target_map_dir)

    # Build dict to tokenize target
    target_token_map = defaultdict(dict)
    for i, (_, values) in enumerate(target_map.items()):
        for j, (short, _) in enumerate(values):
            target_token_map[i][short] = j

    bad_words = []
    feature_map = {}
    features = []
    targets = []
    feature_i = 0
    for word in data:
        try:
            assert len(word.get("postag", "")) == 9
            # Build features
            feature = []
            norm = word["norm"]
            syllables = syllabify(norm)
            for syllable in syllables:
                if syllable not in feature_map:
                    feature_map[syllable] = feature_i
                    feature_i += 1
                token = feature_map[syllable]
                feature.append(token)
            # Build targets
            target = []
            postag = word["postag"]
            for i, tag in enumerate(postag):
                match (i, tag):
                    case ("5", "d"):  # Treat depondent verbs as medio-passive
                        i, tag = ("5", "e")
                token = target_token_map[i][tag]
                target.append(token)
            # Append to data
            features.append(feature)
            targets.append(target)
        except (AssertionError, KeyError):
            bad_words.append(word)

    length_match = len(features) == len(targets)
    assert (
        length_match
    ), f"Feature and target lengths ({len(features), len(targets)}) do not match."

    logger.info("Success! Tokenized features and targets.")

    # Export
    export_pkl(feature_map, feature_map_dir)
    export_pkl(features, features_dir)
    export_pkl(targets, targets_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    get_postag_map()
    tokenize()
