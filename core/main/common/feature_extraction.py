import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from core.utils import config as c


def feature_extraction(df: pd.DataFrame, is_bidirectional: bool, test_size=0.3):
    if is_bidirectional:
        x_train, x_test, y_train, y_test = train_test_split(df['preprocessed'], df['bidirectional_statement'],
                                                            test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = train_test_split(df['preprocessed'], df['statement'], test_size=test_size)

    # Encoding the Labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)  # Training Set
    y_test = encoder.fit_transform(y_test)  # Test Set

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['preprocessed'])

    x_train_tfidf = vectorizer.transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    # Export to file
    f = open("../../../files/core/vectorization/vocabulary_index.json", "w")
    json.dump(vectorizer.vocabulary_, f)

    return [x_train_tfidf, x_test_tfidf, y_train, y_test]


feature_extraction(c.training_set, True)
