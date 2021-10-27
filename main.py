import math

import dataIngestion

# Executable
import predictionModel
import scoringModel
import testModel
import wordSchemaProcessingModel

if __name__ == '__main__':
    trainingData = dataIngestion.readDataFrame("dataset/train2.tsv")
    testData = dataIngestion.readDataFrame("dataset/test2.tsv")

    #print(scoringModel.combainedScoreModel(trainingData))
    testModel.testAccuracy(trainingData, testData)

