import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


def sentence_processing(training: pd.DataFrame, testing: pd.DataFrame, model: str):
    y_train_bi, y_test_bi = encoder.fit_transform(training['bidirectional_statement']), encoder.fit_transform(testing['bidirectional_statement'])
    y_train_hex, y_test_hex = encoder.fit_transform(training['statement']), encoder.fit_transform(testing['statement'])

    def labeller(arg: str):
        switcher = {"s": "lstm_subjects", "sj": "sj_combined", "sm": "sm_combined", "smj": "smj_combined"}
        return switcher.get(arg, "n/a")

    x_train, x_test = training[labeller(model)], testing[labeller(model)]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(x_train))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train, x_test = pad_sequences(x_train), pad_sequences(x_test)

    return [x_train, x_test, y_train_bi, y_test_bi, y_train_hex, y_test_hex]
