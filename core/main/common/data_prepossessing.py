import nltk
import pandas as pd
from nltk import WordNetLemmatizer

from core.utils import config as c
from core.utils import stop_words as s

pd.set_option("max_rows", 600)


# Dimensional Managing
def bidirectional_labels(label: str):
    if label == c.labels[0] or label == c.labels[1] or label == c.labels[2]:
        return "true"
    else:
        return "false"


# Data Preprocessing
def statement_data_preprocessing(corpus: pd.DataFrame, export_path: str, file_name: str):
    def bagOfWord(sentence: str):
        # Delete the punctuations in the sentence
        tokenizer = nltk.RegexpTokenizer(r"\w+")  # Regex Format
        tokenizedSentence = tokenizer.tokenize(sentence.lower())
        meaningfulTokenSet, wnl = [], WordNetLemmatizer()  # Word Lemmatization

        for w in tokenizedSentence:
            if w not in s.ENGLISH_STOP_WORDS and w.isalpha():
                meaningfulTokenSet.append(wnl.lemmatize(w))

        return meaningfulTokenSet

    corpus['subjects'].dropna(inplace=True)  # Remove any blank rows
    corpus['preprocessed'] = [bagOfWord(entry) for entry in corpus['subjects']]  # Tokenization
    corpus['bidirectional_statement'] = [bidirectional_labels(label) for label in corpus['statement']]  # Bi-directional Label

    corpus.to_csv(f"{export_path}{file_name}", index=True,
                  columns=['subjects', 'preprocessed', 'statement', 'bidirectional_statement'])

    return corpus[['preprocessed', 'statement', 'bidirectional_statement']]


statement_data_preprocessing(c.training_data, "../../../files/core/preprocessed/", "training_set.csv")
