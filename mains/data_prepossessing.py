import pandas as pd
from utils import dataIngestion
from algorithm import vectorization as v

pd.set_option("max_rows", 600)

directory_path = "../files/texts"
trainingData = dataIngestion.readDataFrame("../dataset/train2.tsv")


# Unique Words Extractor
def save_unique_words_to_file(df: pd.DataFrame, path: str, statement: str):
    word_list, counter = [], 0

    for _, rows in df.T.items():
        if rows['statement'] == statement:
            bog = v.bagOfWord(rows['subjects'])
            for w in bog:
                if w not in word_list:
                    word_list.append(w)

            counter += 1

    try:
        full_path = path + statement + ".txt"
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
# save_unique_words_to_file(trainingData, "../files/indexes")

# Get All Sentences from the dataset
# save_sentences_to_file(trainingData, "../files/word_list")

# Get All Sentences from the dataset
# save_sentences_to_one_file(trainingData, "../files/indexes")
