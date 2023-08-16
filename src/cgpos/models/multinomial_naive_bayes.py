# -*- coding: utf-8 -*-
import logging
import math
from collections import Counter, defaultdict
from typing import Collection

from cgpos.utils.util import flatten


def ngrams(sequence: Collection, ngram_range: (int, int)) -> list:
    """
    Return (1, n) n-grams for input sequence.

    Arguments
    - sequence: Sequence of tokens.
    - n: Depth of n-grams to generate.
    """
    start, end = ngram_range
    grams = []
    len_sequence = len(sequence)
    for i in range(start, end + 1):
        n_passes = len_sequence - i + 1
        if n_passes >= 1:
            for j in range(n_passes):
                gram = tuple(sequence[j : (i + j)])
                grams.append(gram)
        else:
            break

    return grams


def count_vectors(sequence: Collection, ngram_range: (int, int)) -> Counter:
    """
    Return count vectors of n-gram bag-of-syllables.

    Argument
    - words: Dictionary of word syllables.
    - var: Variable to count.
    - n: Depth of n-grams to generate.
    """
    counts = Counter()
    for gram in ngrams(sequence, ngram_range):
        counts[gram] += 1
    return counts


class MultinomialNaiveBayes:
    """
    Implement Multinomial Naive Bayes with Laplace smoothing and N-gram range.
    """

    def __init__(self, alpha=1.0, ngram_range=(1, 1)):
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.log_likelihoods = None
        self.log_priors = None
        self.feature_counts = None
        self.class_counts = None
        self.classes = None
        self.V = None

    def fit(self, X: Collection, y: Collection):
        X_grams = [count_vectors(word, self.ngram_range) for word in X]
        N = len(y)
        V = len(set(flatten(X_grams)))
        classes = set(y)
        class_counts = Counter()
        feature_counts = defaultdict(Counter)
        for i in range(N):
            class_i = y[i]
            features_i = X_grams[i]
            class_counts[class_i] += 1
            feature_counts[class_i].update(features_i)

        def set_default_factory(value):
            return lambda: value

        log_priors = {key: math.log(value / N) for key, value in class_counts.items()}
        log_likelihoods = defaultdict(defaultdict)
        for class_i in classes:
            feature_total = sum(feature_counts[class_i].values())
            denominator = feature_total + self.alpha * V
            for key, value in feature_counts[class_i].items():
                numerator = value + self.alpha
                log_likelihood = math.log(numerator / denominator)
                log_likelihoods[class_i][key] = log_likelihood
            laplace = math.log(self.alpha / denominator)
            log_likelihoods[class_i].default_factory = set_default_factory(laplace)

        self.V = V
        self.classes = classes
        self.class_counts = class_counts
        self.feature_counts = feature_counts
        self.log_priors = log_priors
        self.log_likelihoods = log_likelihoods

    def predict(self, X: Collection) -> list:
        X_grams = [ngrams(word, self.ngram_range) for word in X]
        preds = []
        for word in X_grams:
            probs = self.log_priors.copy()
            for gram in word:
                for class_i in self.classes:
                    probs[class_i] += self.log_likelihoods[class_i][gram]
            max_prob = float("-inf")
            argmax = None
            for class_i, prob in probs.items():
                if prob > max_prob:
                    max_prob = prob
                    argmax = class_i
            preds.append(argmax)

        return preds


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    pass
