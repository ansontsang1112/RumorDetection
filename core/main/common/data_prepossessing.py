import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from core.utils import config as c
from core.utils import stop_words as s

pd.set_option("max_rows", 600)


# Dimensional Managing
def bidirectional_labels(label: str):
    if label == c.labels[0] or label == c.labels[1] or label == c.labels[2]:
        return "true"
    else:
        return "false"


def bagOfWord(sentence: str):
    # Delete the punctuations in the sentence
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # Regex Format
    tokenizedSentence = tokenizer.tokenize(sentence.lower())
    meaningfulTokenSet, wnl = [], WordNetLemmatizer()  # Word Lemmatization

    for w in tokenizedSentence:
        if w not in s.ENGLISH_STOP_WORDS and w.isalpha():
            meaningfulTokenSet.append(wnl.lemmatize(w))

    return meaningfulTokenSet


def wordSeparation(listOfStrings: str):
    return listOfStrings.split(",")


def feature_esncoder(feature: str):
    encoder = LabelEncoder()
    return


# Data Preprocessing
def data_preprocessing(corpus: pd.DataFrame, export_path: str, file_name: str):
    corpus['subjects'].dropna(inplace=True)  # Remove any blank rows subjects
    corpus['context'].dropna(inplace=True)  # Remove any blank rows in context

    corpus['preprocessed'] = [bagOfWord(entry) for entry in corpus['subjects']]  # Tokenization of Subjects
    corpus['context_preprocessed'] = [bagOfWord(entry) for entry in corpus['context']]  # Tokenization of Justification

    corpus['bidirectional_statement'] = [bidirectional_labels(label) for label in
                                         corpus['statement']]  # Bi-directional Label

    corpus['metadata_1_aspect'], corpus['metadata_2_speaker'] = [wordSeparation(entry) for entry in corpus['aspect']], LabelEncoder().fit_transform(corpus['speaker'])
    corpus['sj_feature'] = corpus['preprocessed'] + corpus['context_preprocessed']

    corpus.to_csv(f"{export_path}{file_name}", index=True,
                  columns=['subjects', 'preprocessed', 'context', 'context_preprocessed', 'metadata_1_aspect', 'metadata_2_speaker',
                           'sj_feature', 'statement', 'bidirectional_statement'])

    return corpus[['subjects', 'preprocessed', 'context', 'metadata_1_aspect', 'metadata_2_speaker',
                   'context_preprocessed', 'sj_feature', 'statement', 'bidirectional_statement']]


data_preprocessing(c.training_data, "../../../files/core/preprocessed/", "training_set_preprocessed.csv")
data_preprocessing(c.testing_data, "../../../files/core/preprocessed/", "testing_set_preprocessed.csv")
