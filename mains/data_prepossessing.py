import pandas as pd
from nltk import WordNetLemmatizer

from utils import dataIngestion
from algorithm import vectorization as v

pd.set_option("max_rows", 600)

directory_path = "../files/texts"
trainingData = dataIngestion.readDataFrame("../dataset/train2.tsv")
testingData = dataIngestion.readDataFrame("../dataset/test2.tsv")


# Unique Words Extractor
def save_unique_words_to_file(df: pd.DataFrame, path: str, isTraining: bool):
    word_list = []
    wnl = WordNetLemmatizer()

    statements = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']

    for statement in statements:
        for _, rows in df.T.items():
            if rows['statement'] == statement:
                bog = v.bagOfWord(rows['subjects'])
                for w in bog:
                    s = wnl.lemmatize(w)
                    if s not in word_list:
                        word_list.append(s)

        try:
            if isTraining:
                full_path = path + statement + "_training.txt"
            else:
                full_path = path + statement + "_testing.txt"

            f = open(full_path, 'w', encoding='utf-8')
            for word in word_list:
                f.write(word + "\n")
            f.close()

            print(f"Unique Words for '{statement}' built at {path}")

        except Exception as e:
            print("write error: " + e)


# Sentences Extractor
def save_sentences_to_file(df: pd.DataFrame, path: str):
    sentences, counter = [], 0

    for _, rows in df.T.items():
        f = open(path + "/" + rows['id'] + ".txt", 'w', encoding='utf-8')
        f.write(rows['subjects'])
        f.close()

        counter += 1

    print(f"Sentence Files was built at {path}")


# Sentences Extractor
def save_sentences_to_one_file(df: pd.DataFrame, path: str):
    sentences, counter = [], 0
    f = open(path + "/" + "sentence.txt", 'a', encoding='utf-8')

    for _, rows in df.T.items():
        f.write(rows['subjects'] + "\n")
        counter += 1
        print(f"{counter} / {len(df)}")

    f.close()


# Get Unique Words from the dataset
save_unique_words_to_file(trainingData, "../files/indexes/word_list/", True)
save_unique_words_to_file(testingData, "../files/indexes/word_list/", False)

# Get All Sentences from the dataset
# save_sentences_to_file(trainingData, "../files/word_list")

# Get All Sentences from the dataset
# save_sentences_to_one_file(trainingData, "../files/indexes")
