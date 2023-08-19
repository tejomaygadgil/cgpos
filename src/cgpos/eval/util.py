"""
This module contains utilities for building and evaluating part-of-speech model.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

from itertools import product

import numpy as np
from omegaconf import DictConfig

from cgpos.utils.path import export_pkl


def ngram_range_grid(ngram_depth: int) -> list:
    """
    Generate a parameter grid from (1, 1) to (1, ngram_depth).
    """
    ngram_range = []
    for i in range(1, ngram_depth):
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
