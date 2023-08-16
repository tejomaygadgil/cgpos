# -*- coding: utf-8 -*-
import logging
import math
from collections import Counter, defaultdict
from typing import Collection

from cgpos.utils.util import (
    flatten,
)


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


class multinomial_naive_bayes:
    def __init__(self, alpha=1, ngram_range=(1, 1)):
        self.log_likelihoods = None
        self.log_priors = None
        self.feature_counts = None
        self.class_counts = None
        self.n_classes = None
        self.V = None
        self.alpha = 1
        self.ngram_range = ngram_range

    def fit(self, X, y):
        X_grams = [count_vectors(word, self.ngram_range) for word in X]
        N = len(y)
        V = len(set(flatten(X_grams)))
        n_classes = max(y) + 1
        class_counts = [0] * n_classes
        feature_counts = defaultdict(Counter)
        for i in range(N):
            class_i = y[i]
            features_i = X_grams[i]
            class_counts[class_i] += 1
            feature_counts[class_i].update(features_i)

        log_priors = [
            math.log(class_counts[class_i] / N) for class_i in range(n_classes)
        ]

        def create_default_factory(val):
            return lambda: val

        log_likelihoods = defaultdict(defaultdict)
        for class_i in range(n_classes):
            feature_total = sum(feature_counts[class_i].values())
            denominator = feature_total + self.alpha * V
            for key, value in feature_counts[class_i].items():
                numerator = value + self.alpha
                log_likelihood = math.log(numerator / denominator)
                log_likelihoods[class_i][key] = log_likelihood
            laplace = math.log(self.alpha / denominator)
            log_likelihoods[class_i].default_factory = create_default_factory(laplace)

        self.V = V
        self.n_classes = n_classes
        self.class_counts = class_counts
        self.feature_counts = feature_counts
        self.log_priors = log_priors
        self.log_likelihoods = log_likelihoods

    def predict(self, X):
        X_grams = [ngrams(word, self.ngram_range) for word in X]
        preds = []
        for x in X_grams:
            probs = self.log_priors.copy()
            for gram in x:
                for class_i in range(self.n_classes):
                    probs[class_i] += self.log_likelihoods[class_i][gram]
            max_prob = float("-inf")
            argmax = None
            for i, prob in enumerate(probs):
                if prob > max_prob:
                    max_prob = prob
                    argmax = i
            preds.append(argmax)

        return preds


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # # Train test split
    # random.seed(seed)
    # shuffled = random.sample(data, len(data))
    # train_ind = int(len(data) * train)
    # train_data = shuffled[:train_ind]
    # test_data = shuffled[train_ind:]
    #
    # if verbose:
    #     logger.info(
    #         f"Train-test split {train}: (train={len(train_data)}, test={len(test_data)}) [seed={seed}]"
    #     )

    # Calculate metrics
    # cat2int = import_pkl(config.perseus.cat2int)
    # int2cat = import_pkl(config.perseus.int2cat)
    # true = np.array([cat2int[(var, label)] for label in true_labels])
    # pred = np.array([cat2int.get((var, label), 0) for label in pred_labels])
    # labels = [int2cat[(var, i)] for i in np.unique(true)]
    #
    # metrics = {
    #     "accuracy": sum(true == pred) / len(test_data),
    #     "classification_report": classification_report(true, pred, target_names=labels),
    #     "confusion": pd.DataFrame(
    #         confusion_matrix(true, pred), index=labels, columns=labels
    #     ),
    # }
    #
    # if verbose:
    #     logger.info(
    #         f"Accuracy: {float(metrics['accuracy']) * 100:.2f}%. Classification report:\n"
    #         + metrics["classification_report"]
    #     )

    pass
