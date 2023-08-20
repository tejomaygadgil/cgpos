"""
This module contains utilities for building and evaluating part-of-speech model.
"""
import os
from collections import defaultdict

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>
from itertools import product
from typing import re

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix

from cgpos.util.path import export_pkl, import_pkl


def ngram_range_grid(ngram_depth: int) -> list:
    """
    Generate a parameter grid from (1, 1) to (1, ngram_depth).
    """
    ngram_range = []
    for i in range(1, ngram_depth + 1):
        ngram_range.append((1, i))

    return ngram_range


def get_clf_args(clf_param: DictConfig) -> list:
    """
    Generate Cartesian product of classifier parameters for tuning.
    """
    param_grid = {}
    if "alpha" in clf_param:
        start = clf_param.alpha.start
        stop = clf_param.alpha.stop
        step = clf_param.alpha.step
        param_grid["alpha"] = np.arange(start=start, stop=stop, step=step)
    if "ngram_range" in clf_param:
        depth = clf_param.ngram_range.depth
        param_grid["ngram_range"] = ngram_range_grid(depth)
    if "ngram_depth" in clf_param:
        start = clf_param.ngram_depth.start
        stop = clf_param.ngram_depth.stop
        param_grid["ngram_depth"] = list(range(start, stop + 1))
    # Generate grid
    clf_args = []
    params = list(param_grid.keys())
    param_product = product(*[param_grid[key] for key in params])
    for output in param_product:
        clf_arg = {}
        for i, param in enumerate(params):
            clf_arg[param] = output[i]
        clf_args.append(clf_arg)

    return clf_args


def run_clf(i: int, clf_arg: dict, run_clf_arg: dict):
    """
    Utility to run model in parallel.
    """
    # Unpack args
    clf_method = run_clf_arg["clf_method"]
    f1_score = run_clf_arg["f1_score"]
    f1_average = run_clf_arg["f1_average"]
    X_train = run_clf_arg["X_i_train"]
    y_train = run_clf_arg["y_i_train"]
    X_dev = run_clf_arg["X_i_dev"]
    y_dev = run_clf_arg["y_i_dev"]
    score_dir_stem = run_clf_arg["score_dir_stem"]
    pred_dir_stem = run_clf_arg["score_dir_stem"]
    export_pred = run_clf_arg["export_pred"]
    # Get score
    clf = clf_method(**clf_arg)
    y_pred = clf.fit(X_train, y_train).predict(X_dev)
    score = f1_score(y_pred, y_dev, average=f1_average)
    # Export
    score_dir = score_dir_stem + f"{i}.pkl"
    export_pkl(score, score_dir, verbose=False)
    if export_pred:
        pred_dir = pred_dir_stem + f"{i}.pkl"
        export_pkl(y_pred, pred_dir, verbose=False)


def get_run_data(test_dir):
    """
    Get train and test data for a CV test fold.
    """
    X_train_dir = os.path.join(test_dir, "X_train.pkl")
    X_train = import_pkl(X_train_dir)
    X_test_dir = os.path.join(test_dir, "X_test.pkl")
    X_test = import_pkl(X_test_dir)
    y_train_dir = os.path.join(test_dir, "y_train.pkl")
    y_train = import_pkl(y_train_dir)
    y_test_dir = os.path.join(test_dir, "y_test.pkl")
    y_test = import_pkl(y_test_dir)

    return X_train, X_test, y_train, y_test


def get_scores(scores_dir: str) -> defaultdict:
    """
    Import CV scores for a training run.
    """
    scores = defaultdict(lambda: defaultdict(dict))
    scores_name = os.listdir(scores_dir)
    for score_name in scores_name:
        score_dir = os.path.join(scores_dir, score_name)
        score = import_pkl(score_dir, verbose=False)

        # Parse file name
        args_dict = {}
        args = os.path.splitext(score_name)[0].split("_")
        for arg in args:
            key, value = re.match(pattern=r"([a-zA-Z]+)(\d+)", string=arg).groups()
            value = int(value)
            args_dict[key] = value
        clf = args_dict["clf"]
        target = args_dict["target"]
        tune = args_dict["tune"]
        clfarg = args_dict["clfarg"]

        # Store score
        clf_key = (clf, clfarg)
        scores[target][clf_key][tune] = score

    return scores


def eval_scores(scores: defaultdict) -> defaultdict:
    """
    Evaluate scores for each fold of a training run.
    """
    target_eval = defaultdict(defaultdict)
    for target, clf_keys in scores.items():
        for clf_key, tunes in clf_keys.items():
            values = list(tunes.values())
            result = sum(values) / len(values)

            # Store result
            target_eval[target][clf_key] = result

    return target_eval


def find_best_model(
    target_eval: defaultdict, param_grids: list, clfs_name: list, targets_name: list
) -> dict:
    """
    Find best model parameters from scores of a training run.
    """
    tagger_args = {}
    for target, clfargs in target_eval.items():
        best_clfkey = max(clfargs, key=clfargs.get)
        best_clf, best_clfarg = best_clfkey
        best_clf_name = clfs_name[best_clf]
        best_param = param_grids[best_clf][best_clfarg]
        tagger_args[targets_name[target]] = (
            best_clf_name,
            best_param,
        )

    return tagger_args


def get_report_contents(
    y_pred: np.array,
    y_test: np.array,
    pp_tagger_args: str,
    targets_name: list,
    targets_long: list,
) -> str:
    """
    Generate contents of eval report.
    """
    # Generate report
    classification_reports = []
    confusion_matrices = []
    targets_len = len(targets_name)
    for i in range(targets_len):
        y_pred_i = y_pred[:, i]
        y_test_i = y_test[:, i]
        target_long = targets_long[i]
        target_len = list(range(len(target_long)))
        classification_report_i = classification_report(
            y_pred_i,
            y_test_i,
            labels=target_len,
            target_names=target_long,
        )
        confusion_matrix_i = pd.DataFrame(
            confusion_matrix(y_pred_i, y_test_i),
            index=target_long,
            columns=target_long,
        ).to_markdown()
        classification_reports.append(classification_report_i)
        confusion_matrices.append(confusion_matrix_i)

    report_content = "\n".join(
        [
            pp_tagger_args,
            "\n".join(classification_reports),
            "\n".join(confusion_matrices),
        ]
    )

    return report_content
