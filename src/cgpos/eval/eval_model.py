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
        tagger_args = find_best_model(target_eval, param_grids, clfs_name, targets_name)
        pp_tagger_args = pprint.pformat(tagger_args)
        logger.info(f"Best model parameters: \n{pp_tagger_args}")

        # Build best model
        tagger_clfs = {}
        for target_name, (clf_name, clf_arg) in tagger_args.items():
            clf_method = getattr(clf_module, clf_name)
            tagger_clf = clf_method(**clf_arg)
            tagger_clfs[target_name] = tagger_clf
        tagger = PartOfSpeechTagger(targets_name, tagger_clfs)

        # Calculate accuracy
        y_pred = tagger.fit(X_train, y_train).predict(X_test)
        accuracy = np.mean((y_pred == y_test).all(axis=1))
        print(f"Best model accuracy: {(accuracy * 100):.2f}%")

        # Export model
        tagger_dir = os.path.join(config.eval.models_dir, "tagger_cv.pkl")
        export_pkl(tagger, tagger_dir)

        # Generate report
        report_content = get_report_contents(
            y_pred, y_test, pp_tagger_args, targets_name, targets_long
        )
        file_path = get_abs_dir("reports/report.txt")
        with open(file_path, "w") as file:
            file.write(report_content)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    eval_model()
