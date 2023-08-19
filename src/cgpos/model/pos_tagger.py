"""
This module provides a multiouput part-of-speech tagger to use on prediction data.
"""
import logging
import pprint

import numpy as np
from tqdm import tqdm

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>


class PartOfSpeechTagger:
    """
    Predicts fine-grained part-of-speech.
    """

    def __init__(self, targets_name, clfs):
        self.targets_name = targets_name
        self.clfs = clfs
        pass

    def __str__(self):
        return f"Part of Speech tagger: \n{pprint.pformat(self.clfs)}"

    def fit(self, X: list, y: np.array):
        len_targets = len(self.targets_name)
        assert len_targets == y.shape[1], "y is wrong shape."
        assert len(X) == y.shape[0], "X and y should have same the number of rows."
        logger = logging.getLogger(__name__)
        logger.info("Fitting Part of Speech tagger:")

        # Fit all models
        for target in tqdm(range(len_targets)):
            y_i = y[:, target]
            target_name = self.targets_name[target]
            clf = self.clfs[target_name]
            clf.fit(X, y_i)

        # Set attributes
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X: list) -> list:
        # Get predictions
        preds = []
        len_targets = len(self.targets_name)
        for target in range(len_targets):
            target_name = self.targets_name[target]
            clf = self.clfs[target_name]
            pred = clf.predict(X)
            preds.append(pred)
        preds = np.array(preds).T

        return preds
