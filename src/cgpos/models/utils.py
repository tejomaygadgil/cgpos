"""
This module contains utilities for building and evaluating part-of-speech models.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from collections import Counter
from typing import Collection, Union

from sklearn.model_selection import StratifiedShuffleSplit


def ngrams(sequence: list, n: Union[tuple, int]) -> Collection:
    """
    Return (1, n) n-grams for input sequence.

    Arguments
    - sequence: Sequence of tokens.
    - n: N-gram range, or depth of n-grams to generate.
    """
    match n:
        case tuple():
            start, end = n
            grams = []
            len_sequence = len(sequence)
            for i in range(start, end + 1):
                n_passes = len_sequence - i + 1
                if n_passes >= 1:
                    for j in range(n_passes):
                        gram = tuple(sequence[j : (i + j)])
                        grams.append(gram)
            return grams

        case int():
            len_sequence = len(sequence)
            n_passes = max(1, len_sequence - n + 1)
            return [tuple(sequence[i : (i + n)]) for i in range(n_passes)]


def count_vectors(sequence: list, ngram_range: (int, int)) -> Counter:
    """
    Return count vectors of n-gram bag-of-syllables.

    Arguments
    - words: Dictionary of word syllables.
    - var: Variable to count.
    - n: Depth of n-grams to generate.
    """
    counts = Counter()
    for gram in ngrams(sequence, ngram_range):
        counts[gram] += 1
    return counts


def stratified_shuffle(X, y, clf, return_preds, sss_args, clf_args):
    """
    Return scores of stratified shuffle CV.

    Arguments:
    - X: Features.
    - y: Target.
    - clf: Classifier.
    - return_preds: Boolean to return predictions.
    - sss_args: Additional arguments for StratifiedShuffleSplit.
    - clf_args: Additional arguments for clf.
    """

    sss = StratifiedShuffleSplit(**sss_args)

    scores = []
    dummy_X = [0] * len(y)
    splits = sss.split(dummy_X, y)
    for train_indices, test_indices in splits:
        X_train = [X[index] for index in train_indices]
        X_test = [X[index] for index in test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        classifier = clf(**clf_args)
        score = classifier.fit(X_train, y_train).score(X_test, y_test, return_preds)
        scores.append(score)

    return scores
