"""
Ths module implements Multinomial Naive Bayes for part-of-speech tagging.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

from cgpos.models.utils import count_vectors, ngrams
from cgpos.utils.util import flatten


class Classifier(ABC):
    """
    Abstract base class for classifiers.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Trains classifier to predict y using X.
        Arguments
        - X: features.
        - y: targets.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Generate prediction for y using X.
        Arguments
        - X: features.
        """
        pass

    def score(self, X, y, return_preds=False):
        """
        Return accuracy of predictions of X as compared to y.
        Arguments
        X: features.
        y: targets.
        """
        len_y = len(y)
        y_pred = self.predict(X)
        num_correct = [y_pred[i] == y[i] for i in range(len_y)]
        accuracy = sum(num_correct) / len_y

        if return_preds:
            return accuracy, y_pred
        return accuracy


class MultinomialNaiveBayes(Classifier):
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
        name = "Multinomial Naive Bayes"
        return f"{name} with alpha={self.alpha} and ngram_range={self.ngram_range}"

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
        self.X_ = X
        self.y_ = y
        self.V_ = V
        self.classes_ = classes
        self.class_counts_ = class_counts
        self.feature_counts_ = feature_counts
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


class StupidBayes(Classifier):
    """
    Implement Stupid Bayes that just adds things up.

    Arguments
    - n: Maximum n-gram depth.
    """

    def __init__(self, ngram_depth=1):
        self.ngram_depth = ngram_depth

    def __str__(self):
        name = "Stupid Bayes"
        return f"{name} with ngram_depth={self.ngram_depth}"

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

        # Get feature counts
        self.gram_counts_ = gram_counts

        return self

    def predict(self, X: list) -> list:
        def _ngram_backoff(sequence, gram_dict, n):
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
                    sub_y_dist = _ngram_backoff(gram, gram_dict, n - 1)
                    dist.update(sub_y_dist)

            return dist

        # Get predictions
        preds = []
        for x in X:
            pred = None
            y_dist = _ngram_backoff(x, self.gram_counts_, self.n)
            if y_dist:
                pred = max(y_dist, key=y_dist.get)
            preds.append(pred)

        return preds
