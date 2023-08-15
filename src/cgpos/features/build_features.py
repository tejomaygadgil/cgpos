# -*- coding: utf-8 -*-
import logging

import hydra
from omegaconf import DictConfig

from cgpos.utils.util import export_pkl, import_pkl


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def tokenize(config: DictConfig):
    """
    Tokenize features and build target.
    """
    logger = logging.getLogger(__name__)
    logger.info("Tokenizing Perseus features and targets:")

    # Import directories
    import_dir = config.data.cleaned
    # Export directories
    feature_map_dir = config.reference.feature_map
    features_dir = config.data.features

    # Import data
    _, data, _ = import_pkl(import_dir)

    # Build features
    feature_map = {}
    features = []
    feature_token = 0
    for syllables in data:
        feature = []
        for syllable in syllables:
            if syllable not in feature_map:
                feature_map[syllable] = feature_token
                feature_token += 1
            token = feature_map[syllable]
            feature.append(token)
        features.append(feature)

    assert len(features) == len(data), "Input and output lengths do not match."

    # Export
    export_pkl(feature_map, feature_map_dir)
    export_pkl(features, features_dir)

    logger.info(
        f"Success! Extracted {len(feature_map)} tokens from {len(features)} words."
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    tokenize()
