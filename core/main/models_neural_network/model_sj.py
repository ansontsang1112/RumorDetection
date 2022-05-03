import time

from core.utils import config as c
from core.main.evaluation import time_complexity as t
from core.main.common import neural_network_modelling as n
from core.main.evaluation import model_evaluation as e
from core.main.common import feature_extraction_nn as f


def lstm(in_dim):
    x_train, x_test, y_train_bi, y_test_bi, y_train_hex, y_test_hex = f.sentence_processing(c.training_set,
                                                                                            c.testing_set, "sj")
    t1 = time.time()

    model, model_hex = n.RNN(in_dim), n.RNN(in_dim)

    model.fit(x_train, y_train_bi, validation_data=(x_test, y_test_bi), epochs=3, batch_size=64)
    model_hex.fit(x_train, y_train_hex, validation_data=(x_test, y_test_hex), epochs=3, batch_size=64)

    t2 = time.time()
    t.time_complexity(t1, t2, "LSTM")

    # Final evaluation of the model
    e.eva_nn(model, model_hex, x_test, y_test_bi, y_test_hex, "LSTM")
