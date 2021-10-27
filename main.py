# Executable
from algorithm import vectorization
from model import wordSchemaProcessingModel, scoringModel
import dataIngestion

if __name__ == '__main__':

    trainingData = dataIngestion.readDataFrame("dataset/train2.tsv")
    testData = dataIngestion.readDataFrame("dataset/test2.tsv")

    # print(scoringModel.combainedScoreModel(trainingData))
    # testModel.testAccuracy(trainingData, testData)

    print(scoringModel.combainedScoreModel(trainingData))


