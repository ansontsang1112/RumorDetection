# Executable
import json

from waste import scoringModel
from utils import dataIngestion

if __name__ == '__main__':

    trainingData = dataIngestion.readDataFrame("../dataset/train2.tsv")
    testData = dataIngestion.readDataFrame("../dataset/test2.tsv")

    # print(scoringModel.combainedScoreModel(trainingData))
    # testModel.testAccuracy(trainingData, testData)

    # Export to file
    file = open("feature_extracted_result_TF_IDF.txt", "w")
    json.dump(scoringModel.combinedScoreModel(trainingData), file)
    file.close()


