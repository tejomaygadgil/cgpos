"""
Config values.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

# PRE-TRAINING


# FINE-TUNING
# Import dirs
ft_raw_data = "data/raw/treebank_data-master/v1.6/greek/data"
ft_raw_targets_map = "data/raw/treebank_data-master/v2.0/Greek/TAGSETS.xml"

# Save dirs
ft_processed = "data/interim/ft_raw.pkl"
ft_targets_map = "data/reference/ft_targets_map.pkl"
ft_normalized = "data/interim/ft_normalized.pkl"
ft_cleaned = "data/processed/ft_cleaned.pkl"
ft_targets = "data/processed/ft_targets.pkl"
