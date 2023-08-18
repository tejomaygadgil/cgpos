"""
This module contains utilities for building and evaluating part-of-speech models.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from collections import Counter
from typing import Collection, Union

from cgpos.utils.util import export_pkl


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


def ngram_range_grid(ngram_depth):
    """
    Generate a parameter grid of all combinations from (1, 1) to (ngram_depth, ngram_depth).

    Arguments
    - ngram_depth: Maximum depth of ngram range grid.
    """
    ngram_range = []
    for i in range(1, ngram_depth):
        for j in range(i, ngram_depth):
            ngram_range.append((i, j))

    return ngram_range


def run_clf(clfarg_i, clf_arg, run_clf_arg):
    """
    Utility to run model in parallel.
    """
    # Unpack args
    clf = run_clf_arg["clf"]
    f1_score = run_clf_arg["f1_score"]
    X_i_train = run_clf_arg["X_i_train"]
    y_i_train = run_clf_arg["y_i_train"]
    X_i_dev = run_clf_arg["X_i_dev"]
    y_i_dev = run_clf_arg["y_i_dev"]
    f1_average = run_clf_arg["f1_average"]
    target_i = run_clf_arg["target_i"]
    test_i = run_clf_arg["test_i"]
    tune_i = run_clf_arg["tune_i"]

    # Get score
    clf_i = clf(**clf_arg)
    y_i_pred = clf_i.fit(X_i_train, y_i_train).predict(X_i_dev)
    score = f1_score(y_i_pred, y_i_dev, average=f1_average)
    # Export
    export_path = f"data/results/test_{test_i}_target_{target_i}_tune_{tune_i}_clfarg_{clfarg_i}_score.pkl"
    export_pkl(score, export_path, verbose=False)
