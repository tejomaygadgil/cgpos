"""
Config values.
"""
import datetime

# Author: Tejomay Gadgil <tejomaygadgil@gmail.com>

# PRE-TRAINING
# Directories
pt_dir = "data/raw/diorisis"
# Data
pt_beta = "data/interim/pt_beta.pkl"
pt_uni = "data/interim/pt_uni.pkl"
pt_norm = "data/processed/pt_norm.pkl"
pt_syl = "data/processed/pt_syl.pkl"
pt_syl_cloud = "/content/drive/MyDrive/Colab Notebooks/pt_syl.pkl"
# Train
pt_params = "data/train/pt_params.pkl"
pt_stoi = "data/train/pt_stoi.pkl"
pt_itos = "data/train/pt_itos.pkl"
pt_train = "data/train/pt_train"
pt_val = "data/train/pt_val"
# Model
pt_checkpoint = "/content/drive/MyDrive/Colab Notebooks/pt_checkpoint_"

# FINE-TUNING
# Directories
ft_dir = "data/raw/treebank_data-master/v1.6/greek/data"
ft_targets_map_dir = "data/raw/treebank_data-master/v2.0/Greek/TAGSETS.xml"
# Data
ft_targets_map = "data/reference/ft_targets_map.pkl"
ft_raw = "data/interim/ft_raw.pkl"
ft_clean = "data/interim/ft_clean.pkl"
ft_norm = "data/processed/ft_norm.pkl"
ft_targets = "data/processed/ft_targets.pkl"
ft_syl = "data/processed/ft_syl.pkl"
# Train
ft_params = "data/train/ft_params.pkl"
# Model
ft_checkpoint = "/content/drive/MyDrive/Colab Notebooks/ft_checkpoint.tar"
