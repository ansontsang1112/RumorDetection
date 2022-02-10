import pandas as pd

from algorithm import vectorization


def scoring(string: str):
    score = {
        "true": 1,
        "mostly-true": 0.7,
        "half-true": 0.5,
        "barely-true": 0.3,
        "false": 0.2,
        "pants-fire": 0
    }
    return score[string]


def wordScoringModel(rowData: {}):
    # Take the row data from Dataset
    statementScore = scoring(rowData['statement'].lower())
    sentenceSet = vectorization.bagOfWord(rowData['subjects'])
    scoreDict = {}

    for word in sentenceSet:
        scoreDict[word] = statementScore

    return scoreDict


def combainedScoreModel(inputDataFrame: pd.DataFrame):
    fullyScoringModel, uniqueness, processCounter = {}, [], 0

    for _, row in inputDataFrame.T.items():
        requestHeader = {}
        sentence, statement = row['subjects'], row['statement']
        requestHeader['subjects'], requestHeader['statement'] = sentence, statement
        singleSentenceScore = wordScoringModel(requestHeader)
        vectorized_data = vectorization.tf_idf_vectorizer(requestHeader['subjects'], inputDataFrame)

        for word in singleSentenceScore:
            if word in fullyScoringModel:
                fullyScoringModel[word] += (singleSentenceScore[word] * vectorized_data[word])
            else:
                fullyScoringModel[word] = (singleSentenceScore[word] * vectorized_data[word])
                uniqueness.append(word)

        processCounter += 1
        print(f"Modeling Process: {processCounter} / {len(inputDataFrame)}")


    return fullyScoringModel
