import math

import predictionModel
import scoringModel


def testAccuracy(trainSet, testSet):
    resultSet = {}
    scoreModel = scoringModel.combainedScoreModel(trainSet)

    for _, row in testSet.T.items():
        testID, testStatement, testSentence = row['id'], row['statement'], row['subjects']

        resultSet[testID] = {}
        resultSet[testID]["score"] = predictionModel.predictByWordScore(scoreModel, testSentence)
        resultSet[testID]["real"] = testStatement
        resultSet[testID]["logistic"] = 1 / (1 + math.pow(math.e, (-resultSet[testID]["score"])))
        print(resultSet[testID])
