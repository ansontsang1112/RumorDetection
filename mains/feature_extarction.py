import glob
import json
import os
from pathlib import Path
from mains import data_prepossessing as d

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import mains.data_prepossessing
from utils import dataIngestion, stop_words

trainingData = dataIngestion.readDataFrame("../dataset/train2.tsv")
testingData = dataIngestion.readDataFrame("../dataset/test2.tsv")


# Import as sentences (Legacy Code)
# sentence_list = [y[1] for y in list(enumerate([x.strip() for x in open("../files/indexes/sentence.txt", "r", encoding='utf-8')]))]

# Import as word filter (Legacy Code)
# word_list = [y[1] for y in list(enumerate([x.strip() for x in open("../files/indexes/word_list/word.txt", "r", encoding='utf-8')]))]


def tf_idf_by_statement(statement: str, df: pd.DataFrame, export: bool, isTraining: bool):
    extracted_id_list, paths, export_path, text_files = [], "../files/texts/", "../files/out/", []

    if isTraining:
        extension = "_training"
    else:
        extension = "_testing"

    if len(os.listdir("../files/texts")) == 0:
        d.save_sentences_to_file(df, "../files/texts")

    if not Path(f"../files/indexes/word_list/{statement}{extension}.txt").is_file():
        d.save_unique_words_to_file(df, "../files/indexes/word_list/", statement, isTraining)

    for _, rows in df.T.items():
        if rows['statement'] == statement:
            extracted_id_list.append(rows['id'])
            text_files.append(f"{paths}{rows['id']}.txt")

    text_titles = [Path(text).stem for text in text_files]

    vectorizer = TfidfVectorizer(input='filename', stop_words=stop_words.ENGLISH_STOP_WORDS)
    vector = vectorizer.fit_transform(text_files)

    tfidf_df = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out(), index=text_titles)
    result_series = tfidf_df.sum().sort_values(ascending=True).to_dict()

    if export:
        tfidf_df.to_csv(f"{export_path}csv/{statement}{extension}_TFIDF.csv", index=True)

        file = open(f"{export_path}word_result_TF_IDF/{statement}{extension}_tfidf.txt", "w")
        json.dump(result_series, file)
        file.close()

        print(f"TDIDF CSV and Sum Text for '{statement}{extension}' built at {export_path}csv and {export_path}word_result_TF_IDF")


tf_idf_by_statement("true", trainingData, True, True)
tf_idf_by_statement("mostly-true", trainingData, True, True)
tf_idf_by_statement("half-true", trainingData, True, True)
tf_idf_by_statement("barely-true", trainingData, True, True)
tf_idf_by_statement("false", trainingData, True, True)
tf_idf_by_statement("pants-fire", trainingData, True, True)
tf_idf_by_statement("true", testingData, True, False)
tf_idf_by_statement("mostly-true", testingData, True, False)
tf_idf_by_statement("half-true", testingData, True, False)
tf_idf_by_statement("barely-true", testingData, True, False)
tf_idf_by_statement("false", testingData, True, False)
tf_idf_by_statement("pants-fire", testingData, True, False)