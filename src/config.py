"""
Config values.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

# PRE-TRAINING
# Raw data dir
pt_raw = "data/raw/diorisis"
pt_processed = "data/interim/pt_raw.pkl"
pt_text = "data/processed/pt_text.pkl"
pt_syl = "data/processed/pt_syl.pkl"

# FINE-TUNING
# Raw data dir
ft_raw = "data/raw/treebank_data-master/v1.6/greek/data"
ft_raw_targets_map = "data/raw/treebank_data-master/v2.0/Greek/TAGSETS.xml"
# Output dir
ft_processed = "data/interim/ft_raw.pkl"
ft_targets_map = "data/reference/ft_targets_map.pkl"
ft_normalized = "data/interim/ft_normalized.pkl"
ft_text = "data/processed/ft_text.pkl"
ft_targets = "data/processed/ft_targets.pkl"
ft_syl = "data/processed/ft_syl.pkl"
