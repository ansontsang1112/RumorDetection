from keras import Input, Model, Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout, concatenate, Flatten
import keras.backend as K
from core.utils import config as c


def RNN(in_dim, output_layer_activation='sigmoid', loss='binary_crossentropy', optimizer='adam'):
    print(f"Starting RNN")

    model = Sequential()
    model.add(Embedding(input_dim=in_dim, output_dim=c.out_dim))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation=output_layer_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', f1_score_nn])

    return model


def f1_score_nn(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
