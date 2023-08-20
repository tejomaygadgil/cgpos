"""
Contains utility functions to read data and clean Greek text.
"""

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

import logging
import os
import pickle
from pathlib import Path


def get_abs_dir(path):
    """
    Helper function to return the absolute path from within a package (data, features, etc.)
    """
    abs_path = os.path.join(Path(__file__).resolve().parents[3], path)
    return abs_path


def import_pkl(path, verbose=True):
    logger = logging.getLogger(__name__)
    path = get_abs_dir(path)
    if verbose:
        logger.info(f"Importing {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def export_pkl(data, path, verbose=True):
    logger = logging.getLogger(__name__)
    path = get_abs_dir(path)
    if verbose:
        logger.info(f"Exporting {path}")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]
