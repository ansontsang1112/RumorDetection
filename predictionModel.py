import nltk
import pandas as pd


def predictByWordScore(model: {}, sentence: str):
    score, index = 0, 0

    # Sentence Preprocessing
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # Regex Format
    tokenizedSentence = tokenizer.tokenize(sentence)

    for word in tokenizedSentence:
        if word in model:
            score += model[word]
            index += 1
        else:
            index = 1

    return score / index
