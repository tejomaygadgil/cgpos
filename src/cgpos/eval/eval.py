"""
This module selects the best model and evaluates it on the test set.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os
import pprint
from importlib import import_module

import hydra
import numpy as np
from omegaconf import DictConfig

from cgpos.eval.util import (
    eval_scores,
    find_best_model,
    get_report_contents,
    get_run_data,
    get_scores,
)
from cgpos.model.pos_tagger import PartOfSpeechTagger
from cgpos.util.path import export_pkl, get_abs_dir, import_pkl


@hydra.main(config_path="../../../config", config_name="config", version_base=None)
def eval_model(config: DictConfig):
    """
    Select best model from training results.
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model:")

    # Load run (latest if not specified)
    runs_dir = get_abs_dir(config.runs.runs_dir)
    run = config.runs.run
    if not run:
        runs = next(os.walk(runs_dir))[1]
        latest_run = sorted(runs)[-1]
        run = latest_run
    run_dir = os.path.join(runs_dir, run)

    # Import data
    targets_name, _, targets_long = import_pkl(config.reference.targets_map)
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
        X_train, X_test, y_train, y_test = get_run_data(test_dir)

        # Import scores
        scores = get_scores(scores_dir)

        # Evaluate scores
        target_eval = eval_scores(scores)

        # Find best model
        pos_tagger_args = find_best_model(
            target_eval, param_grids, clfs_name, targets_name
        )
        pp_pos_tagger_args = pprint.pformat(pos_tagger_args)
        logger.info(f"Best model parameters: \n{pp_pos_tagger_args}")

        # Fit best model
        best_model_clfs = {}
        for target_name, (clf_name, clf_arg) in pos_tagger_args.items():
            clf_method = getattr(clf_module, clf_name)
            best_model_clf = clf_method(**clf_arg)
            best_model_clfs[target_name] = best_model_clf
        best_model = PartOfSpeechTagger(targets_name, best_model_clfs)
        best_model.fit(X_train, y_train)

        # Calculate train and test accuracy
        logger.info("Evaluating best model:")
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        accuracy_train = np.mean((y_pred_train == y_train).all(axis=1))
        accuracy_test = np.mean((y_pred_test == y_test).all(axis=1))
        logger.info(f"Train accuracy: {(accuracy_train * 100):.2f}%")
        logger.info(f"Test accuracy: {(accuracy_test * 100):.2f}%")

        # Export model
        logger.info("Exporting model")
        best_model_dir = config.model
        export_pkl(best_model, best_model_dir)

        # Generate report
        report_content = get_report_contents(
            y_pred_test, y_test, pp_pos_tagger_args, targets_name, targets_long
        )
        file_path = get_abs_dir("reports/report.txt")
        with open(file_path, "w") as file:
            file.write(report_content)
        logger.info(f"Exported report to {file_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    eval_model()
