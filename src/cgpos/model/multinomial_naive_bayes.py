"""
Ths module implements Multinomial Naive Bayes for part-of-speech tagging.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import math
from collections import Counter, defaultdict

from cgpos.model.util import count_vectors, ngrams
from cgpos.util.path import flatten


class MultinomialNaiveBayes:
    """
    Implement Multinomial Naive Bayes with Laplace smoothing and N-gram range.

    Arguments
    - alpha: Laplace/Lidstone smoothing parameter.
    - ngram_range: Range of n-grams to generate.
    """

    def __init__(self, alpha=1, ngram_range=(1, 1)):
        self.alpha = alpha
        self.ngram_range = ngram_range

    def __str__(self):
        return f"Multinomial Naive Bayes with alpha={self.alpha} and ngram_range={self.ngram_range}"

    def __repr__(self):
        return (
            f"MultinomialNaiveBayes(alpha={self.alpha}, ngram_range={self.ngram_range})"
        )

    def fit(self, X: list, y: list):
        # Check parameters
        assert 0.0 <= self.alpha <= 1.0, "alpha should be a float between 0.0 and 1.0."
        assert (
            (type(self.ngram_range) == tuple)
            and (self.ngram_range[1] >= self.ngram_range[0])
            and [value > 0 for value in self.ngram_range]
        ), "Invalid ngram_range."

        # Generate class and feature counts
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

        # Calculate prior and likelihood
        log_priors = {key: math.log(value / N) for key, value in class_counts.items()}
        log_likelihoods = defaultdict(dict)
        log_likelihoods_defaults = defaultdict()
        for class_i in classes:
            feature_total = sum(feature_counts[class_i].values())
            denominator = feature_total + self.alpha * V
            for key, value in feature_counts[class_i].items():
                numerator = value + self.alpha
                log_likelihood = math.log(numerator / denominator)
                log_likelihoods[class_i][key] = log_likelihood
            default_likelihood = self.alpha / denominator
            default_log_likelihood = math.log(default_likelihood)
            log_likelihoods_defaults[class_i] = default_log_likelihood

        # Set attributes
        self.classes_ = classes
        self.log_priors_ = log_priors
        self.log_likelihoods_ = log_likelihoods
        self.log_likelihoods_defaults_ = log_likelihoods_defaults

        return self

    def predict(self, X: list) -> list:
        X_grams = [ngrams(word, self.ngram_range) for word in X]
        preds = []
        for word in X_grams:
            probs = self.log_priors_.copy()
            for gram in word:
                for class_i in self.classes_:
                    default_log_likelihood = self.log_likelihoods_defaults_[class_i]
                    probs[class_i] += self.log_likelihoods_[class_i].get(
                        gram, default_log_likelihood
                    )
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

    Arguments
    - n: Maximum n-gram depth.
    """

    def __init__(self, ngram_depth=1):
        self.ngram_depth = ngram_depth

    def __str__(self):
        return f"Stupid Bayes with ngram_depth={self.ngram_depth}"

    def __repr__(self):
        return f"StupidBayes(ngram_depth={self.ngram_depth})"

    def fit(self, X: list, y: list):
        # Check parameters
        assert (type(self.ngram_depth) == int) and (
            self.ngram_depth >= 0
        ), "n should be int >= 0."

        # Generate n-grams
        ngram_range = (1, self.ngram_depth)
        X_ngrams = [ngrams(x, ngram_range) for x in X]

        # Count
        gram_counts = defaultdict(Counter)
        for i, x_ngrams in enumerate(X_ngrams):
            y_i = y[i]
            for ngram in x_ngrams:
                gram_counts[ngram][y_i] += 1

        classes = set(y)
        sorted_classes = sorted(classes)
        n_classes = len(classes)
        middle_n = int(n_classes / 2)
        default_pred = sorted_classes[middle_n]

        # Set attributes
        self.classes_ = classes
        self.gram_counts_ = gram_counts
        self.default_pred_ = default_pred

        return self

    def predict(self, X: list) -> list:
        def ngram_backoff(sequence, gram_dict, n):
            """
            Recursively look up n-grams in gram_dict.
            """
            if n == 0:
                return Counter()

            dist = Counter()
            sequence_ngrams = ngrams(sequence, n)
            for gram in sequence_ngrams:
                if gram in gram_dict:
                    dist.update(gram_dict[gram])
                else:
                    sub_dist = ngram_backoff(gram, gram_dict, n - 1)
                    dist.update(sub_dist)

            return dist

        # Get predictions
        preds = []
        for x in X:
            y_dist = ngram_backoff(x, self.gram_counts_, self.ngram_depth)
            if y_dist:
                pred = max(y_dist, key=y_dist.get)
            else:
                pred = self.default_pred_
            preds.append(pred)

        return preds
