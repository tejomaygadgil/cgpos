"""
Trains part-of-speech tagger on data features.
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

from cgpos.models.util import get_clf_args, run_clf
from cgpos.utils.util import get_abs_dir, import_pkl


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """
    Train model with shuffled CV and stratified shuffled CV for hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info("Tokenizing Perseus features:")

    # Import data
    uid, text, targets = import_pkl(config.data.cleaned)
    features = import_pkl(config.data.features)
    target_names, target_short, target_long = import_pkl(config.reference.target_map)

    # Set export dir
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = get_abs_dir(f"data/results/{timestamp}")
    # Create the directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set data
    X = features
    y = np.array(targets)

    # Get clf
    clf_module = import_module(config.train.clf_module)
    clf_name = config.train.clf
    clf = getattr(clf_module, clf_name)
    logger.info(f"Training {clf_name}:")

    # Get CV args
    eval_split_args = config.train.eval_split
    tune_split_args = config.train.tune_split
    f1_average = config.train.f1_average

    # Set up parameter grid
    clf_param = config.param_grid[clf_name]
    clf_args = get_clf_args(clf_param)

    # Eval CV loop
    ss = ShuffleSplit(**eval_split_args)
    dummy_X = [0] * len(y)
    eval_splits = ss.split(dummy_X, y)
    for eval_i, (_temp_indices, _test_indices) in enumerate(eval_splits):
        logger.info(f"Test split {eval_i + 1} of {eval_split_args['n_splits']}:")
        # X_test = [X[index] for index in test_indices]
        # y_test = y[test_indices]

        _X_temp = [X[index] for index in _temp_indices]
        _y_temp = y[_temp_indices]

        # Target loop
        targets_len = len(target_names)
        for target_i in range(targets_len):
            logger.info(
                f"Target {target_i + 1} of {targets_len} ({target_names[target_i]}):"
            )
            _y_i_temp = _y_temp[:, target_i]

            # Hyperparameter tuning loop
            sss = StratifiedShuffleSplit(**tune_split_args)
            defaultdict(lambda: defaultdict(list))
            dummy_X = [0] * len(_y_i_temp)
            tune_splits = sss.split(dummy_X, _y_i_temp)
            for tune_i, (train_indices, dev_indices) in enumerate(tune_splits):
                logger.info(
                    f"Tune split {tune_i + 1} of {tune_split_args['n_splits']}:"
                )
                X_i_train = [_X_temp[index] for index in train_indices]
                y_i_train = _y_i_temp[train_indices]

                X_i_dev = [_X_temp[index] for index in dev_indices]
                y_i_dev = _y_i_temp[dev_indices]

                file_stem = f"/eval_{eval_i}_target_{target_i}_tune_{tune_i}_clfarg_"
                export_dir_stem = os.path.join(results_dir, file_stem)
                run_clf_arg = {
                    "clf": clf,
                    "f1_score": f1_score,
                    "X_i_train": X_i_train,
                    "y_i_train": y_i_train,
                    "X_i_dev": X_i_dev,
                    "y_i_dev": y_i_dev,
                    "f1_average": f1_average,
                    "export_dir_stem": export_dir_stem,
                }

                # Parallelize model runs
                with ProcessPoolExecutor() as executor:
                    futures = []
                    for i, clf_arg in enumerate(clf_args):
                        future = executor.submit(run_clf, i, clf_arg, run_clf_arg)
                        future.append(future)

                    for future in tqdm(futures, total=len(clf_args)):
                        future.result()


if __name__ == "__main__":
    # Log
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    train_model()
