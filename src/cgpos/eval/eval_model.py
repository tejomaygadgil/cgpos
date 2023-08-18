"""
This module selects the best model and evaluates it on the test set.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os
import pprint
import re
from collections import defaultdict

import hydra
from omegaconf import DictConfig

from cgpos.utils.path import get_abs_dir, import_pkl


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
    param_grid_dir = os.path.join(run_dir, "param_grid.pkl")
    param_grid = import_pkl(param_grid_dir)
    clf_name_dir = os.path.join(run_dir, "clf_name.pkl")
    clf_name = import_pkl(clf_name_dir)

    # Iterate over tests
    tests = [name for name in os.listdir(run_dir) if "test_" in name]
    for test in tests:
        test_dir = os.path.join(run_dir, test)
        scores_dir = os.path.join(test_dir, "scores")

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
            target = args_dict["target"]
            tune = args_dict["tune"]
            clfarg = args_dict["clfarg"]

            # Store score
            scores[target][clfarg][tune] = score

        # Evaluate scores
        target_eval = defaultdict(defaultdict)
        for target, clfargs in scores.items():
            for clfarg, tunes in clfargs.items():
                values = list(tunes.values())
                result = sum(values) / len(values)

                # Store result
                target_eval[target][clfarg] = result

        # Find best model
        best_models = {}
        for target, clfargs in target_eval.items():
            best_clfarg = max(clfargs, key=clfargs.get)
            best_models[targets_name[target]] = param_grid[best_clfarg]

        logger.info(f"Best parameters for {clf_name}: \n{pprint.pformat(best_models)}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    eval_model()
