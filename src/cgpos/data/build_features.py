"""
Includes classes and functions to build model features from cleaned data.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging

import hydra
from omegaconf import DictConfig

from cgpos.util.path import export_pkl, import_pkl


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def tokenize(config: DictConfig):
    """
    Tokenize features.
    """
    logger = logging.getLogger(__name__)
    logger.info("Tokenizing Perseus features:")

    # Import directories
    import_dir = config.data.cleaned
    # Export directories
    features_map_dir = config.reference.features_map
    features_dir = config.data.features

    # Import data
    _, data, _ = import_pkl(import_dir)

    # Build features
    features_map = {}
    features = []
    feature_token = 0
    for syllables in data:
        feature = []
        for syllable in syllables:
            if syllable not in features_map:
                features_map[syllable] = feature_token
                feature_token += 1
            token = features_map[syllable]
            feature.append(token)
        feature_tuple = tuple(feature)
        features.append(feature_tuple)

    assert len(features) == len(data), "Input and output lengths do not match."

    # Export
    export_pkl(features_map, features_map_dir)
    export_pkl(features, features_dir)

    logger.info(
        f"Success! Extracted {len(features_map)} tokens from {len(features)} words."
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    tokenize()
