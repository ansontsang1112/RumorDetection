import math

import pandas as pd
import nltk
import multiprocessing
from utils import stop_words as s


def bagOfWord(sentence: str):
    # Delete the punctuations in the sentence
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # Regex Format
    tokenizedSentence = tokenizer.tokenize(sentence.lower())
    meaningfulTokenSet = []

    for sentFormatting in tokenizedSentence:
        if sentFormatting not in s.ENGLISH_STOP_WORDS:
            meaningfulTokenSet.append(sentFormatting)

    return meaningfulTokenSet


def tf_vectorization(token: []) -> {}:
    # TF(term) = Frequency of a ‘term’ appears in document  / total number of words in the document
    tf_score, frequency, uniqueness = {}, {}, []

    for word in token:
        if word in uniqueness:
            frequency[word] += 1
        else:
            frequency[word] = 1
            uniqueness.append(word)

    for word in frequency:
        tf_score[word] = frequency[word] / len(token)

    return tf_score


def idf_vectorization(token: [], dataFrame: pd.DataFrame) -> {}:
    # IDF(term) = log(Total Number of Docs / number of Docs with 'term' in it)
    document_size, idf_score, frequency, uniqueness = len(dataFrame), {}, {}, []

    for _, row in dataFrame.T.items():
        raw_sentence, in_uniqueness = row['subjects'], []

        def sentence_preprocessing(sentence: str):
            return bagOfWord(sentence)

        processed_sentence = sentence_preprocessing(raw_sentence)

        for word in token:
            if word in processed_sentence:
                if word in uniqueness:
                    if word not in in_uniqueness:
                        frequency[word] += 1
                else:
                    frequency[word] = 0
                    uniqueness.append(word)
                    in_uniqueness.append(word)

    for word in frequency:
        if frequency[word] == 0:
            idf_score[word] = 0
        else:
            idf_score[word] = math.log(document_size / frequency[word])

    return idf_score


def tf_idf_vectorizer(raw_data: str, dataFrame: pd.DataFrame) -> {}:
    token = bagOfWord(raw_data)

    TFIDF_score, TF_score, IDF_score = {}, tf_vectorization(token), idf_vectorization(token, dataFrame)

    processes = []

    def tf_idf() -> {}:
        # TF-IDF = TF(term) x IDF(term)
        TFIDF_score[word] = TF_score[word] * IDF_score[word]

    for word in token:
        p = multiprocessing.Process(target=tf_idf(), args=(word,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    return TFIDF_score
