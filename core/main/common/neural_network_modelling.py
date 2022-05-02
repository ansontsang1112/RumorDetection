from keras import Input, Model, Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout, concatenate, Flatten
import keras.backend as K
from core.utils import config as c


def RNN_MI(hidden_layer_activation='relu', output_layer_activation='sigmoid', loss='binary_crossentropy',
           optimizer='adam'):
    print(f"Starting RNN (Multi-Input)")

    # Handling Inputs
    input1, input2, input3 = Input(shape=[c.max_len]), Input(shape=[c.max_len]), Input(shape=[c.max_len])

    # First Operation
    x = Dense(64, activation=hidden_layer_activation)(input1)
    x = Dense(32, activation=hidden_layer_activation)(input1)
    x = Dense(4, activation=hidden_layer_activation)(input1)
    x = Model(input1, x)

    # Second Operation
    y = Dense(4, activation=hidden_layer_activation)(input2)
    y = Model(input2, y)

    # Second Operation
    z = Dense(4, activation=hidden_layer_activation)(input3)
    z = Model(input3, z)

    combined = concatenate([x.output, y.output, z.output])

    layer = Embedding(c.MAX_FEATURES, c.out_put_dim)(combined)
    layer = LSTM(64)(layer)
    layer = Dense(2, name='FC1')(layer)
    layer = Activation(hidden_layer_activation)(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation(output_layer_activation)(layer)

    model = Model(inputs=[x.output, y.output, z.output], outputs=layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', f1_score_nn])
    return model


def RNN(output_layer_activation='softmax', loss='binary_crossentropy', optimizer='adam'):
    print(f"Starting RNN")

    model = Sequential()
    model.add(Embedding(input_dim=c.in_dim, output_dim=c.out_dim))
    model.add(LSTM(100))
    model.add(Dense(1, activation=output_layer_activation))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', f1_score_nn])

    return model


def f1_score_nn(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
