import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def feature_extraction_s(training: pd.DataFrame, testing: pd.DataFrame, val: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, x_val, y_train, y_test, y_val = training['preprocessed'], testing['preprocessed'], val[
            'preprocessed'], training[
                                                             'bidirectional_statement'], testing[
                                                             'bidirectional_statement'], val['bidirectional_statement']
    else:
        x_train, x_test, x_val, y_train, y_test, y_val = training['preprocessed'], testing['preprocessed'], val[
            'preprocessed'], training['statement'], \
                                                         testing['statement'], val['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set
    y_val = encoder.fit_transform(y_val)  # Val Set

    # TF-IDF Vectorization
    vectorizer_training, vectorizer_testing, vectorizer_val = TfidfVectorizer(), TfidfVectorizer(), TfidfVectorizer()
    vectorizer_training.fit(training['preprocessed'])
    vectorizer_testing.fit(testing['preprocessed'])
    vectorizer_val.fit(val['preprocessed'])

    x_train_tfidf = vectorizer_training.transform(x_train)
    x_test_tfidf = vectorizer_testing.transform(x_test)
    x_val_tfidf = vectorizer_val.transform(x_val)

    return [x_train_tfidf, x_test_tfidf, x_val_tfidf, y_train, y_test, y_val]


def feature_extraction_sj(training: pd.DataFrame, testing: pd.DataFrame, val: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, x_val, y_train, y_test, y_val = training['sj_feature'], testing['sj_feature'], val['sj_feature'], training[
            'bidirectional_statement'], testing['bidirectional_statement'], val['bidirectional_statement']
    else:
        x_train, x_test, x_val, y_train, y_test, y_val = training['sj_feature'], testing['sj_feature'], val['sj_feature'], training['statement'], \
                                           testing['statement'], val['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set
    y_val = encoder.fit_transform(y_val)

    # TF-IDF Vectorization
    vectorizer_training, vectorizer_testing, vectorizer_val = TfidfVectorizer(), TfidfVectorizer(), TfidfVectorizer()
    vectorizer_training.fit(training['sj_feature'])
    vectorizer_testing.fit(testing['sj_feature'])
    vectorizer_val.fit(val['sj_feature'])

    x_train_tfidf = vectorizer_training.transform(x_train)
    x_test_tfidf = vectorizer_testing.transform(x_test)
    x_val_tfidf = vectorizer_val.transform(x_val)

    return [x_train_tfidf, x_test_tfidf, x_val_tfidf, y_train, y_test, y_val]


def feature_extraction_sm(training: pd.DataFrame, testing: pd.DataFrame, val: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, x_val, y_train, y_test, y_val = training[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           testing[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           val[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           training['bidirectional_statement'], testing['bidirectional_statement'], val['bidirectional_statement']
    else:
        x_train, x_test, x_val, y_train, y_test, y_val = training[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           testing[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           val[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           training['statement'], testing['statement'], val['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set
    y_val = encoder.fit_transform(y_val)

    # TF-IDF Vectorization
    pipeline = ColumnTransformer([('tdidf_subjects', TfidfVectorizer(), 'preprocessed'),
                                  ('tdidf_aspect', TfidfVectorizer(), 'metadata_1_aspect')],
                                 remainder='passthrough')

    x_train_tfidf = pipeline.fit_transform(x_train)
    x_test_tfidf = pipeline.fit_transform(x_test)
    x_val_tfidf = pipeline.fit_transform(x_val)

    return [x_train_tfidf, x_test_tfidf, x_val_tfidf, y_train, y_test, y_val]


def feature_extraction_smj(training: pd.DataFrame, testing: pd.DataFrame, val: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, x_val, y_train, y_test, y_val = training[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                     'metadata_2_speaker']], \
                                           testing[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                    'metadata_2_speaker']], \
                                           val[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                    'metadata_2_speaker']], \
                                           training['bidirectional_statement'], testing['bidirectional_statement'], val['bidirectional_statement']
    else:
        x_train, x_test, x_val, y_train, y_test, y_val = training[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                     'metadata_2_speaker']], \
                                           testing[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                    'metadata_2_speaker']], \
                                           val[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                    'metadata_2_speaker']], \
                                           training['statement'], testing['statement'], val['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set
    y_val = encoder.fit_transform(y_val)

    # TF-IDF Vectorization
    pipeline = ColumnTransformer([('tfidf_subjects', TfidfVectorizer(), 'preprocessed'),
                                  ('tfidf_context', TfidfVectorizer(), 'context_preprocessed'),
                                  ('tfidf_aspect', TfidfVectorizer(), 'metadata_1_aspect')],
                                 remainder='passthrough')

    x_train_tfidf = pipeline.fit_transform(x_train)
    x_test_tfidf = pipeline.fit_transform(x_test)
    x_val_tfidf = pipeline.fit_transform(x_val)

    return [x_train_tfidf, x_test_tfidf, x_val_tfidf, y_train, y_test, y_val]
