"""
This module selects the best model and evaluates it on the test set.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os

import hydra
from omegaconf import DictConfig

from cgpos.utils.path import get_abs_dir, import_pkl


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def eval_model(config: DictConfig):
    """
    Train model with shuffled CV and stratified shuffled CV for hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model:")

    # Load run (latest if not specified)
    runs_dir = get_abs_dir(config.runs_dir)
    run = config.eval.run
    if not run:
        dirs = os.listdir(runs_dir)[1]
        latest_run = sorted(dirs)[-1]
        run = latest_run
    run_dir = os.path.join(runs_dir, run)

    # Import data
    param_grid_dir = os.path.join(run_dir, "param_grid.pkl")
    import_pkl(param_grid_dir)

    # Iterate over tests
    tests = [name for name in os.listdir(run_dir) if "test_" in name]
    for _test in tests:
        pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    eval_model()
