import pandas as pd

import wordSchemaProcessingModel


def scoring(string: str):
    score = {
        "true": 4,
        "mostly-true": 3,
        "half-true": 2,
        "barely-true": 1,
        "false": 0,
        "pants-fire": -1
    }
    return score[string]


def wordScoringModel(rowData: {}):
    # Take the row data from Dataset
    statementScore = scoring(rowData['statement'].lower())
    sentenceSet = wordSchemaProcessingModel.wordProcessingModel(rowData['subjects'])
    scoreDict = {}

    for word in sentenceSet:
        scoreDict[word] = statementScore

    return scoreDict


def combainedScoreModel(inputDataFrame: pd.DataFrame):
    fullyScoringModel = {}

    uniqueness, wordCount = [], {}

    for _, row in inputDataFrame.T.items():
        requestHeader = {}
        sentence, statement = row['subjects'], row['statement']
        requestHeader['subjects'], requestHeader['statement'] = sentence, statement
        singleSentenceScore = wordScoringModel(requestHeader)

        for word in singleSentenceScore:
            if word in fullyScoringModel:
                fullyScoringModel[word] += singleSentenceScore[word]
                wordCount[word] += 1
            else:
                fullyScoringModel[word] = singleSentenceScore[word]
                wordCount[word] = 1
                uniqueness.append(word)

    return fullyScoringModel
