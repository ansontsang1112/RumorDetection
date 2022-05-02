import time

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from core.utils import config as c
from core.main.evaluation import time_complexity as t
from core.main.common import neural_network_modelling as n
from core.main.evaluation import cross_validation as cv

encoder = LabelEncoder()
x_train, x_test = c.training_set['sj_combined'], c.testing_set['sj_combined'],
y_train, y_test = encoder.fit_transform(c.training_set['bidirectional_statement']), encoder.fit_transform(c.training_set['bidirectional_statement'])
y_train_hex, y_test_hex = encoder.fit_transform(c.training_set['statement']), encoder.fit_transform(c.training_set['statement'])

t1 = time.time()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train, x_test = pad_sequences(x_train), pad_sequences(x_test)

model, model_hex = n.RNN(), n.RNN()

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)
model_hex.fit(x_train, y_train_hex, validation_data=(x_test, y_test_hex), epochs=3, batch_size=64)

t2 = time.time()
t.time_complexity(t1, t2, "LSTM")

# Final evaluation of the model
cv.validate(model, model_hex, x_test, y_train, y_test, "LSTM")

