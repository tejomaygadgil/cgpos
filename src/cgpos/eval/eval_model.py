"""
This module selects the best model and evaluates it on the test set.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os
import pprint
import re
from collections import defaultdict
from importlib import import_module

import hydra
import numpy as np
from omegaconf import DictConfig

from cgpos.model.pos_tagger import PartOfSpeechTagger
from cgpos.utils.path import export_pkl, get_abs_dir, import_pkl


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def eval_model(config: DictConfig):
    """
    Select best model from training results.
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model:")

    # Load run (latest if not specified)
    runs_dir = get_abs_dir(config.runs_dir)
    run = config.eval.run
    if not run:
        runs = next(os.walk(runs_dir))[1]
        latest_run = sorted(runs)[-1]
        run = latest_run
    run_dir = os.path.join(runs_dir, run)

    # Import data
    targets_name, _, _ = import_pkl(config.reference.targets_map)
    clf_module = import_module(config.train.clf_module)
    clfs_name = config.train.clfs

    # Import param grids
    param_grids = []
    param_grid_dir = os.path.join(run_dir, "param_grid")
    for clf_name in clfs_name:
        clf_param_grid_dir = os.path.join(param_grid_dir, f"{clf_name}.pkl")
        param_grid = import_pkl(clf_param_grid_dir)
        param_grids.append(param_grid)

    # Iterate over tests
    tests = [name for name in os.listdir(run_dir) if "test_" in name]
    for test in tests:
        test_dir = os.path.join(run_dir, test)
        scores_dir = os.path.join(test_dir, "scores")

        # Import data
        X_train_dir = os.path.join(test_dir, "X_train.pkl")
        X_train = import_pkl(X_train_dir)
        X_test_dir = os.path.join(test_dir, "X_test.pkl")
        X_test = import_pkl(X_test_dir)
        y_train_dir = os.path.join(test_dir, "y_train.pkl")
        y_train = import_pkl(y_train_dir)
        y_test_dir = os.path.join(test_dir, "y_test.pkl")
        y_test = import_pkl(y_test_dir)

        # Import scores
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

        # Evaluate scores
        target_eval = defaultdict(defaultdict)
        for target, clf_keys in scores.items():
            for clf_key, tunes in clf_keys.items():
                values = list(tunes.values())
                result = sum(values) / len(values)

                # Store result
                target_eval[target][clf_key] = result

        # Find best model
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

        logger.info(f"Best model parameters: \n{pprint.pformat(tagger_args)}")

        # Build best model
        clfs = {}
        for target_name, (clf_name, clf_arg) in tagger_args.items():
            clf_method = getattr(clf_module, clf_name)
            clf = clf_method(**clf_arg)
            clfs[target_name] = clf

        tagger = PartOfSpeechTagger(targets_name, clfs)
        y_preds = tagger.fit(X_train, y_train).predict(X_test)

        # Calculate accuracy
        accuracy = np.mean((y_preds == y_test).all(axis=1))
        print(f"Best model accuracy: {accuracy* 100:.2f}%")

        # Export model
        tagger_dir = os.path.join(config.eval.models_dir, "tagger_cv.pkl")
        export_pkl(tagger, tagger_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    eval_model()
