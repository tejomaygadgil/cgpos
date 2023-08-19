"""
This module provides a multiouput part-of-speech tagger to use on prediction data.
"""
import pprint

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

    def fit(self, X: list, y: list):
        # Set attributes
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X: list) -> list:
        preds = None

        return preds
