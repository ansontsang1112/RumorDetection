import glob
from pathlib import Path
from mains import data_prepossessing as d

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import mains.data_prepossessing
from utils import dataIngestion

trainingData = dataIngestion.readDataFrame("../dataset/train2.tsv")

#sentence_list = [y[1] for y in
#                 list(enumerate([x.strip() for x in open("../files/indexes/sentence.txt", "r", encoding='utf-8')]))]


def tf_idf_by_statement(statement: str, df: pd.DataFrame):
    extracted_id_list, paths, text_files = [], "../files/texts/", []

    if not Path(f"../files/indexes/word_list/{statement}.txt").is_file():
        d.save_unique_words_to_file(df, "../files/indexes/word_list/", statement)

    word_list = [y[1] for y in
                 list(enumerate([x.strip() for x in open("../files/indexes/word_list/" + statement + ".txt", "r", encoding='utf-8')]))]

    for _, rows in df.T.items():
        if rows['statement'] == statement:
            extracted_id_list.append(rows['id'])
            text_files.append(f"{paths}/{rows['id']}.txt")

    text_titles = [Path(text).stem for text in text_files]

    vectorizer = TfidfVectorizer(input='filename', stop_words='english')
    vector = vectorizer.fit_transform(text_files)

    tfidf_df = pd.DataFrame(vector.toarray(), index=text_titles, columns=vectorizer.get_feature_names_out())

    return tfidf_df


print(tf_idf_by_statement("true", trainingData))
