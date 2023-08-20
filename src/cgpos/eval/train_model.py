"""
This module trains a model and performs hyperparameter tuning.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from importlib import import_module

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from tqdm import tqdm

from cgpos.eval.util import get_clf_args, run_clf
from cgpos.util.path import export_pkl, get_abs_dir, import_pkl


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def train_model(config: DictConfig):
    """
    Train model with shuffled CV and stratified shuffled CV for hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training model:")

    # Import data
    clfs_name = config.train.clfs
    _, _, targets = import_pkl(config.data.cleaned)
    features = import_pkl(config.data.features)
    targets_name, _, _ = import_pkl(config.reference.targets_map)

    # Set export dir
    runs_dir = get_abs_dir(config.runs.runs_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(runs_dir, timestamp)
    param_grid_dir = os.path.join(run_dir, "param_grid")
    os.makedirs(param_grid_dir)

    # Set data
    X = features
    y = np.array(targets)

    # Get CV args
    clfs_len = len(clfs_name)
    targets_len = len(targets_name)
    test_split_args = config.train.test_split
    tune_split_args = config.train.tune_split
    f1_average = config.train.f1_average
    export_pred = config.train.export_pred

    # Test CV loop
    test_splitter = ShuffleSplit(**test_split_args)
    dummy_X = [0] * len(y)
    test_splits = test_splitter.split(dummy_X, y)
    for test, (_temp_indices, test_indices) in enumerate(test_splits):
        logger.info(f"Test split {test + 1} of {test_split_args['n_splits']}:")
        # Make test dir
        test_dir = os.path.join(run_dir, f"test_{test}")
        scores_dir = os.path.join(test_dir, "scores")
        preds_dir = os.path.join(test_dir, "preds")
        os.makedirs(scores_dir)
        os.makedirs(preds_dir)

        # Make test data
        X_test = [X[index] for index in test_indices]
        y_test = y[test_indices]

        # Export test data
        X_test_export_dir = os.path.join(test_dir, "X_test.pkl")
        y_test_export_dir = os.path.join(test_dir, "y_test.pkl")
        export_pkl(X_test, X_test_export_dir, verbose=False)
        export_pkl(y_test, y_test_export_dir, verbose=False)

        # Make _temp data (to split into train and dev)
        X_temp = [X[index] for index in _temp_indices]
        y_temp = y[_temp_indices]

        # Export as train (to evaluate best model)
        X_train_export_dir = os.path.join(test_dir, "X_train.pkl")
        y_train_export_dir = os.path.join(test_dir, "y_train.pkl")
        export_pkl(X_temp, X_train_export_dir, verbose=False)
        export_pkl(y_temp, y_train_export_dir, verbose=False)

        # Loop through models
        for clf in range(clfs_len):
            # Import clf
            clf_name = clfs_name[clf]
            clf_module = import_module(config.train.clf_module)
            clf_method = getattr(clf_module, clf_name)
            logger.info(f"Training classifier {clf + 1} of {clfs_len} ({clf_name}):")

            # Set up parameter grid
            clf_param = config.param_grid[clf_name]
            clf_param_grid = get_clf_args(clf_param)

            # Export parameter grid
            clf_param_grid_dir = os.path.join(param_grid_dir, f"{clf_name}.pkl")
            export_pkl(clf_param_grid, clf_param_grid_dir)

            # Loop through targets
            for target in range(targets_len):
                target_name = targets_name[target]
                logger.info(f"Target {target + 1} of {targets_len} ({target_name}):")

                # Hyperparameter tuning loop
                _y_i_temp = y_temp[:, target]
                tune_splitter = StratifiedShuffleSplit(**tune_split_args)
                defaultdict(lambda: defaultdict(list))
                dummy_X = [0] * len(_y_i_temp)
                tune_splits = tune_splitter.split(dummy_X, _y_i_temp)
                for tune, (train_indices, dev_indices) in enumerate(tune_splits):
                    logger.info(
                        f"Tune split {tune + 1} of {tune_split_args['n_splits']}:"
                    )

                    # Make train and dev sets
                    X_i_train = [X_temp[index] for index in train_indices]
                    y_i_train = y_temp[:, target][train_indices]

                    X_i_dev = [X_temp[index] for index in dev_indices]
                    y_i_dev = _y_i_temp[dev_indices]

                    # Set run parameters
                    file_stem = f"clf{clf}_target{target}_tune{tune}_clfarg"
                    score_dir_stem = os.path.join(scores_dir, file_stem)
                    pred_dir_stem = os.path.join(preds_dir, file_stem)
                    run_clf_arg = {
                        "clf_method": clf_method,
                        "f1_score": f1_score,
                        "X_i_train": X_i_train,
                        "y_i_train": y_i_train,
                        "X_i_dev": X_i_dev,
                        "y_i_dev": y_i_dev,
                        "f1_average": f1_average,
                        "score_dir_stem": score_dir_stem,
                        "pred_dir_stem": pred_dir_stem,
                        "export_pred": export_pred,
                    }

                    # Parallelize model runs
                    with ProcessPoolExecutor() as executor:
                        futures = []
                        # Loop through parameter grid
                        for i, clf_arg in enumerate(clf_param_grid):
                            future = executor.submit(run_clf, i, clf_arg, run_clf_arg)
                            futures.append(future)

                        for future in tqdm(futures, total=len(clf_param_grid)):
                            future.result()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    train_model()
