import pandas as pd

from core.utils import data_ingestion

training_data = data_ingestion.readDataFrame("../../../dataset/train2.tsv")
testing_data = data_ingestion.readDataFrame("../../../dataset/test2.tsv")

training_set = pd.read_csv("../../../files/core/preprocessed/training_set.csv")
# testing_set = pd.read_csv("../files/core/preprocessed/testing_set.csv")

labels = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
bidirectional_labels = ['true', 'false']
