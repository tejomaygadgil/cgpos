# -*- coding: utf-8 -*-
import logging
import math
from collections import Counter, defaultdict
from typing import Collection, Union

from cgpos.utils.util import flatten


def ngrams(sequence: Collection, n: Union[tuple, int]) -> Collection:
    """
    Return (1, n) n-grams for input sequence.

    Arguments
    - sequence: Sequence of tokens.
    - n: n-gram range, or depth of n-grams to generate.
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
        assert 0 <= alpha <= 1, "0.0 <= alpha <= 1.0."
        assert (type(ngram_range) == tuple) and [
            i > 0 for i in ngram_range
        ], "ngram_range should be non-negative tuple."
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.V = None
        self.classes = None
        self.class_counts = None
        self.feature_counts = None
        self.log_priors = None
        self.log_likelihoods = None
        self.gram_set = None

    def fit(self, X: Collection, y: Collection):
        X_cv = [count_vectors(word, self.ngram_range) for word in X]
        N = len(y)
        V = len(set(flatten(X_cv)))
        classes = set(y)
        class_counts = Counter()
        feature_counts = defaultdict(Counter)
        for i in range(N):
            class_i = y[i]
            features_i = X_cv[i]
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


class StupidBayes:
    """
    Implement Stupid Bayes that just adds things up.
    """

    def __init__(self, n=1):
        assert (type(n) == int) and (n >= 0), "n should be int >= 0."
        self.n = n
        self.gram_counts = None

    def fit(self, X: list, y: list):
        X_ngrams = [ngrams(x, (1, self.n)) for x in X]
        gram_counts = defaultdict(Counter)
        for i, x_ngrams in enumerate(X_ngrams):
            y_i = y[i]
            for ngram in x_ngrams:
                gram_counts[ngram][y_i] += 1
        self.gram_counts = gram_counts

    def predict(self, X: Collection) -> list:
        def _ngram_backoff(sequence, gram_dict, n):
            """
            Recursively looks up n-grams in gram_dict.
            """
            if n == 0:
                return Counter()

            y_dist = Counter()
            sequence_ngrams = ngrams(sequence, n)
            for gram in sequence_ngrams:
                if gram in gram_dict:
                    y_dist.update(gram_dict[gram])
                else:
                    sub_y_dist = _ngram_backoff(gram, gram_dict, n - 1)
                    y_dist.update(sub_y_dist)

            return y_dist

        preds = []
        for x in X:
            y_dist = _ngram_backoff(x, self.gram_counts, self.n)
            pred = None
            if y_dist:
                pred = max(y_dist, key=y_dist.get)
            preds.append(pred)

        return preds


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    pass
