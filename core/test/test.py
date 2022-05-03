import time

import numpy
from keras import Input, Model, Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from core.utils import config as c
from core.main.evaluation import time_complexity as t
from core.main.common import neural_network_modelling as n
from core.main.evaluation import cross_validation as cv

t1 = time.time()

encoder = LabelEncoder()
x_train, x_test = c.training_set['smj_combined'], c.testing_set['smj_combined'],
y_train, y_test = encoder.fit_transform(c.training_set['statement']), encoder.fit_transform(
    c.training_set['statement'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train, x_test = pad_sequences(x_train), pad_sequences(x_test)

model = Sequential()
model.add(Embedding(input_dim=21480, output_dim=32))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', n.f1_score_nn])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

t2 = time.time()
t.time_complexity(t1, t2, "LSTM")

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
print("F1: ", scores[2])
