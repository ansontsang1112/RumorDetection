import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from core.utils import config as c


def feature_extraction_s(training: pd.DataFrame, testing: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, y_train, y_test = training['preprocessed'], testing['preprocessed'], training[
            'bidirectional_statement'], testing['bidirectional_statement']
    else:
        x_train, x_test, y_train, y_test = training['preprocessed'], testing['preprocessed'], training['statement'], \
                                           testing['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set

    # TF-IDF Vectorization
    vectorizer_training, vectorizer_testing = TfidfVectorizer(), TfidfVectorizer()
    vectorizer_training.fit(training['preprocessed'])
    vectorizer_testing.fit(testing['preprocessed'])

    x_train_tfidf = vectorizer_training.transform(x_train)
    x_test_tfidf = vectorizer_testing.transform(x_test)

    # Export to file
    f = open("../../../files/core/vectorization/vocabulary_index_s.json", "w")
    json.dump(vectorizer_training.vocabulary_, f)

    return [x_train_tfidf, x_test_tfidf, y_train, y_test]


def feature_extraction_sj(training: pd.DataFrame, testing: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, y_train, y_test = training['sj_feature'], testing['sj_feature'], training[
            'bidirectional_statement'], testing['bidirectional_statement']
    else:
        x_train, x_test, y_train, y_test = training['sj_feature'], testing['sj_feature'], training['statement'], \
                                           testing['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set

    # TF-IDF Vectorization
    vectorizer_training, vectorizer_testing = TfidfVectorizer(), TfidfVectorizer()
    vectorizer_training.fit(training['sj_feature'])
    vectorizer_testing.fit(testing['sj_feature'])

    x_train_tfidf = vectorizer_training.transform(x_train)
    x_test_tfidf = vectorizer_testing.transform(x_test)

    # Export to file
    f = open("../../../files/core/vectorization/vocabulary_index_sj.json", "w")
    json.dump(vectorizer_training.vocabulary_, f)

    return [x_train_tfidf, x_test_tfidf, y_train, y_test]


def feature_extraction_sm(training: pd.DataFrame, testing: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, y_train, y_test = training[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           testing[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           training['bidirectional_statement'], testing['bidirectional_statement']
    else:
        x_train, x_test, y_train, y_test = training[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           testing[['preprocessed', 'metadata_1_aspect', 'metadata_2_speaker']], \
                                           training['statement'], testing['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set

    # TF-IDF Vectorization
    pipeline = ColumnTransformer([('tdidf_subjects', TfidfVectorizer(), 'preprocessed'),
                                  ('tdidf_aspect', TfidfVectorizer(), 'metadata_1_aspect')],
                                 remainder='passthrough')

    x_train_tfidf = pipeline.fit_transform(x_train)
    x_test_tfidf = pipeline.fit_transform(x_test)

    return [x_train_tfidf, x_test_tfidf, y_train, y_test]


def feature_extraction_smj(training: pd.DataFrame, testing: pd.DataFrame, is_bidirectional: bool):
    if is_bidirectional:
        x_train, x_test, y_train, y_test = training[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                     'metadata_2_speaker']], \
                                           testing[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                    'metadata_2_speaker']], \
                                           training['bidirectional_statement'], testing['bidirectional_statement']
    else:
        x_train, x_test, y_train, y_test = training[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                     'metadata_2_speaker']], \
                                           testing[['preprocessed', 'context_preprocessed', 'metadata_1_aspect',
                                                    'metadata_2_speaker']], \
                                           training['statement'], testing['statement']

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set

    # TF-IDF Vectorization
    pipeline = ColumnTransformer([('tfidf_subjects', TfidfVectorizer(), 'preprocessed'),
                                  ('tfidf_context', TfidfVectorizer(), 'context_preprocessed'),
                                  ('tfidf_aspect', TfidfVectorizer(), 'metadata_1_aspect')],
                                 remainder='passthrough')

    x_train_tfidf = pipeline.fit_transform(x_train)
    x_test_tfidf = pipeline.fit_transform(x_test)

    return [x_train_tfidf, x_test_tfidf, y_train, y_test]
