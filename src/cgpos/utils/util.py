from pathlib import Path


def get_abs_dir(file=__file__):
    """
    Helper function to return the absolute path from within a package (data, features, etc.)
    """
    return Path(file).resolve().parents[3]
