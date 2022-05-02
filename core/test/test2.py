import time

import numpy
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from core.utils import config as c
from core.main.evaluation import time_complexity as t
from core.main.common import neural_network_modelling as n
from core.main.common import feature_extraction as f

x_train, x_test = c.training_set['preprocessed'], c.testing_set['preprocessed'],
y_train, y_test = c.training_set['bidirectional_statement'], c.training_set['bidirectional_statement']

x_f2_train, x_f2_test = c.training_set['metadata_1_aspect'], c.testing_set['metadata_1_aspect']

tokenizer, tokenizer2 = Tokenizer()
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train, x_test = pad_sequences(x_train), pad_sequences(x_test)

model = n.RNN()
model.fit()
print(x_train)


