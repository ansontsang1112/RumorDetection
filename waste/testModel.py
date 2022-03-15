import math

from waste import predictionModel, scoringModel


def testAccuracy(trainSet, testSet):
    resultSet = {}
    scoreModel = scoringModel.combinedScoreModel(trainSet)

    for _, row in testSet.T.items():
        testID, testStatement, testSentence = row['id'], row['statement'], row['subjects']

        resultSet[testID] = {}
        resultSet[testID]["score"] = predictionModel.predictByWordScore(scoreModel, testSentence)
        resultSet[testID]["real"] = testStatement
        resultSet[testID]["logistic"] = 1 / (1 + math.pow(math.e, (-resultSet[testID]["score"])))
        print(resultSet[testID])
