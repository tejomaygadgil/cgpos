"""
This module selects the best model and evaluates it on the test set.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def eval_model(config: DictConfig):
    """
    Train model with shuffled CV and stratified shuffled CV for hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model:")

    # Load run
    runs_dir = config.runs_dir
    run = config.eval.run
    if not run:
        dirs = os.listdir(runs_dir)[1]
        latest_run = sorted(dirs)[-1]
        run = latest_run
    os.path.join(runs_dir, run)

    # # Import data
    # _, _, targets = import_pkl(config.data.cleaned)
    # features = import_pkl(config.data.features)
    # targets_name, _, _ = import_pkl(config.reference.targets_map)
    #
    # # Set export dir
    # current_datetime = datetime.now()
    # timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # runs_dir = get_abs_dir(f"data/runs/{timestamp}")
    # os.makedirs(runs_dir)
    #
    # # Set data
    # X = features
    # y = np.array(targets)
    #
    # # Import clf
    # clf_name = config.train.clf
    # clf_module = import_module(config.train.clf_module)
    # clf = getattr(clf_module, clf_name)
    # logger.info(f"Training {clf_name}:")
    #
    # # Get CV args
    # test_split_args = config.train.test_split
    # tune_split_args = config.train.tune_split
    # f1_average = config.train.f1_average
    # export_pred = config.train.export_pred
    #
    # # Set up parameter grid
    # clf_param = config.param_grid[clf_name]
    # param_grid = get_clf_args(clf_param)
    #
    # # Export parameter grid
    # param_grid_dir = os.path.join(runs_dir, "param_grid.pkl")
    # export_pkl(param_grid, param_grid_dir)
    #
    # # test CV loop
    # test_splitter = ShuffleSplit(**test_split_args)
    # dummy_X = [0] * len(y)
    # test_splits = test_splitter.split(dummy_X, y)
    # for test, (_temp_indices, test_indices) in enumerate(test_splits):
    #     logger.info(f"Test split {test + 1} of {test_split_args['n_splits']}:")
    #     # Make test dir
    #     test_dir = os.path.join(runs_dir, f"test_{test}")
    #     scores_dir = os.path.join(test_dir, "scores")
    #     preds_dir = os.path.join(test_dir, "preds")
    #     os.makedirs(scores_dir)
    #     os.makedirs(preds_dir)
    #
    #     # Make test data
    #     X_test = [X[index] for index in test_indices]
    #     y_test = y[test_indices]
    #
    #     # Export test data
    #     X_test_export_dir = os.path.join(test_dir, "X_test.pkl")
    #     y_test_export_dir = os.path.join(test_dir, "y_test.pkl")
    #     export_pkl(X_test, X_test_export_dir, verbose=False)
    #     export_pkl(y_test, y_test_export_dir, verbose=False)
    #
    #     # Make _temp data (to be split into train and dev)
    #     _X_temp = [X[index] for index in _temp_indices]
    #     _y_temp = y[_temp_indices]
    #
    #     # Export as train (for eval_model)
    #     X_train_export_dir = os.path.join(test_dir, "X_train.pkl")
    #     y_train_export_dir = os.path.join(test_dir, "y_train_.pkl")
    #     export_pkl(X_test, X_train_export_dir, verbose=False)
    #     export_pkl(y_test, y_train_export_dir, verbose=False)
    #
    #     # Loop through targets
    #     targets_len = len(targets_name)
    #     for target in range(targets_len):
    #         target_name = targets_name[target]
    #         logger.info(f"Target {target + 1} of {targets_len} ({target_name}):")
    #
    #         # Hyperparameter tuning loop
    #         _y_i_temp = _y_temp[:, target]
    #         tune_splitter = StratifiedShuffleSplit(**tune_split_args)
    #         defaultdict(lambda: defaultdict(list))
    #         dummy_X = [0] * len(_y_i_temp)
    #         tune_splits = tune_splitter.split(dummy_X, _y_i_temp)
    #         for tune, (train_indices, dev_indices) in enumerate(tune_splits):
    #             logger.info(f"Tune split {tune + 1} of {tune_split_args['n_splits']}:")
    #
    #             # Make train and dev sets
    #             X_i_train = [_X_temp[index] for index in train_indices]
    #             y_i_train = _y_temp[:, target][train_indices]
    #
    #             X_i_dev = [_X_temp[index] for index in dev_indices]
    #             y_i_dev = _y_i_temp[dev_indices]
    #
    #             # Set run parameters
    #             file_stem = f"test_{test}_target_{target}_tune_{tune}_clfarg_"
    #             score_dir_stem = os.path.join(scores_dir, file_stem)
    #             pred_dir_stem = os.path.join(preds_dir, file_stem)
    #             run_clf_arg = {
    #                 "clf": clf,
    #                 "f1_score": f1_score,
    #                 "X_i_train": X_i_train,
    #                 "y_i_train": y_i_train,
    #                 "X_i_dev": X_i_dev,
    #                 "y_i_dev": y_i_dev,
    #                 "f1_average": f1_average,
    #                 "score_dir_stem": score_dir_stem,
    #                 "pred_dir_stem": pred_dir_stem,
    #                 "export_pred": export_pred,
    #             }
    #
    #             # Parallelize model runs
    #             with ProcessPoolExecutor() as executor:
    #                 futures = []
    #                 # Loop through parameter grid
    #                 for i, clf_arg in enumerate(param_grid):
    #                     future = executor.submit(run_clf, i, clf_arg, run_clf_arg)
    #                     futures.append(future)
    #
    #                 for future in tqdm(futures, total=len(param_grid)):
    #                     future.result()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    eval_model()
