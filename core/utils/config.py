from pathlib import Path

import pandas as pd

from core.utils import data_ingestion

training_data = data_ingestion.readDataFrame(f"{Path(__file__).parents[2]}/dataset/train2.tsv")
testing_data = data_ingestion.readDataFrame(f"{Path(__file__).parents[2]}/dataset/test2.tsv")
val_data = data_ingestion.readDataFrame(f"{Path(__file__).parents[2]}/dataset/val2.tsv")

training_set = pd.read_csv(f"{Path(__file__).parents[2]}/files/core/preprocessed/training_set_preprocessed.csv")
testing_set = pd.read_csv(f"{Path(__file__).parents[2]}/files/core/preprocessed/training_set_preprocessed.csv")
val_set = pd.read_csv(f"{Path(__file__).parents[2]}/files/core/preprocessed/val_set_preprocessed.csv")

labels = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
bidirectional_labels = ['true', 'false']

# NN Hyperparameter
MAX_FEATURES = 25000
MAX_WORD = 25000
EMBEDDING_DIM = 100
in_dim = 20480
out_dim = 32

# Cross Validation Hyperparameter
k_fold = 30
