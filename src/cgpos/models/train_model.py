"""
Trains part-of-speech tagger on data features.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from hydra import compose, initialize
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from tqdm import tqdm

from cgpos.models.util import ngram_range_grid
from cgpos.utils.util import export_pkl, import_pkl

if __name__ == "__main__":
    # Load hydra params
    initialize("../../../conf", version_base=None)
    config = compose(config_name="main")

    # Log
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # Import data
    uid, text, targets = import_pkl(config.data.cleaned)
    features = import_pkl(config.data.features)
    target_names, target_short, target_long = import_pkl(config.reference.target_map)

    # Set data
    X = features
    y = np.array(targets)

    # Get args
    clf = config.train.clf
    eval_split_args = config.train.eval_split
    tune_split_args = config.train.tune_split
    f1_average = config.train.f1_average

    # Set up parameter grid
    alpha_min = config.MultinomialNaiveBayes.alpha_min
    alpha_max = config.MultinomialNaiveBayes.max
    alpha_step = config.MultinomialNaiveBayes.alpha_step
    ngram_depth = config.MultinomialNaiveBayes.ngram_depth

    param_grid = {
        "alpha": np.arange(alpha_min, alpha_max, alpha_step),
        "ngram_range": ngram_range_grid(ngram_depth),
    }

    clf_args = []
    param_product = product(param_grid["alpha"], param_grid["ngram_range"])
    for alpha, ngram_range in param_product:
        clf_args.append({"alpha": alpha, "ngram_range": ngram_range})

    # Test CV loop
    ss = ShuffleSplit(**eval_split_args)
    dummy_X = [0] * len(y)
    test_splits = ss.split(dummy_X, y)
    for test_i, (_temp_indices, test_indices) in enumerate(test_splits):
        logging.info(f"Test split {test_i + 1} of {eval_split_args['n_splits']}:")
        X_test = [X[index] for index in test_indices]
        y_test = y[test_indices]

        _X_temp = [X[index] for index in _temp_indices]
        _y_temp = y[_temp_indices]

        # Target loop
        targets_len = len(target_names)
        for target_i in range(targets_len):  # Per target
            logging.info(
                f"Target {target_i + 1} of {targets_len} ({target_names[target_i]}):"
            )
            _y_i_temp = _y_temp[:, target_i]

            # Hyperparameter tuning loop
            sss = StratifiedShuffleSplit(**tune_split_args)
            tune_scores = defaultdict(lambda: defaultdict(list))
            dummy_X = [0] * len(_y_i_temp)
            tune_splits = sss.split(dummy_X, _y_i_temp)
            for tune_i, (train_indices, dev_indices) in enumerate(tune_splits):
                logging.info(
                    f"Tune split {tune_i + 1} of {tune_split_args['n_splits']}:"
                )
                X_i_train = [_X_temp[index] for index in train_indices]
                y_i_train = _y_i_temp[train_indices]

                X_i_dev = [_X_temp[index] for index in dev_indices]
                y_i_dev = _y_i_temp[dev_indices]

                run_clf_arg = {
                    "X_i_train": X_i_train,
                    "y_i_train": y_i_train,
                    "X_i_dev": X_i_dev,
                    "y_i_dev": y_i_dev,
                    "f1_average": f1_average,
                    "target_i": target_i,
                    "test_i": test_i,
                    "tune_i": tune_i,
                }

                def run_clf(clfarg_i, clf_arg, run_clf_arg):
                    """
                    Run classifier with args.
                    """
                    # Unpack args
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

                # Parallelize model runs
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(run_clf, clfarg_i, clf_arg, run_clf_arg)
                        for clfarg_i, clf_arg in enumerate(clf_args)
                    ]

                    for future in tqdm(futures, total=len(clf_args)):
                        future.result()
