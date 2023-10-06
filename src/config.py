"""
Config values.
"""
# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

# PRE-TRAINING
pt_dir = "data/raw/diorisis"
pt_beta = "data/interim/pt_beta.pkl"
pt_uni = "data/interim/pt_uni.pkl"
pt_norm = "data/interim/pt_norm.pkl"
pt_text = "data/processed/pt_text.pkl"
pt_syl = "data/processed/pt_syl.pkl"

# FINE-TUNING
# Directories
ft_dir = "data/raw/treebank_data-master/v1.6/greek/data"
ft_targets_map_dir = "data/raw/treebank_data-master/v2.0/Greek/TAGSETS.xml"
# Data
ft_targets_map = "data/reference/ft_targets_map.pkl"
ft_raw = "data/interim/ft_raw.pkl"
ft_clean = "data/interim/ft_clean.pkl"
ft_text = "data/processed/ft_text.pkl"
ft_targets = "data/processed/ft_targets.pkl"
ft_syl = "data/processed/ft_syl.pkl"
