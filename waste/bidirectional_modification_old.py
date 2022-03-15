import ast
import json

import pandas as pd
from matplotlib import pyplot as plt

path = "../files/bidirectional/"


def bidirectional_dataset_get(direction: bool, export: bool, isTraining: bool):
    input_path = "../files/out/word_result_TF_IDF"
    combined_dict = {}

    if isTraining:
        extension = "_training"
    else:
        extension = "_testing"

    if direction is True:
        file1 = open(f"{input_path}/true{extension}_tfidf.txt")
        file2 = open(f"{input_path}/mostly-true{extension}_tfidf.txt")
        file3 = open(f"{input_path}/half-true{extension}_tfidf.txt")
    else:
        file1 = open(f"{input_path}/barely-true{extension}_tfidf.txt")
        file2 = open(f"{input_path}/false{extension}_tfidf.txt")
        file3 = open(f"{input_path}/pants-fire{extension}_tfidf.txt")

    def dict_export(file):
        contents = file.read()
        dictionary = ast.literal_eval(contents)

        return dictionary

    def dict_combined(data_dict: dict):
        for key in data_dict:
            if key in combined_dict:
                combined_dict[key] += data_dict[key]
            else:
                combined_dict[key] = data_dict[key]

    dict_combined(dict_export(file1))
    dict_combined(dict_export(file2))
    dict_combined(dict_export(file3))

    if export:
        file = open(f"{input_path}raw/{direction}_bidirectional{extension}.txt", "w")
        json.dump(combined_dict, file)
        file.close()

    return combined_dict


def bidirectional_dataset(isTraining: bool):
    data_true, data_false, full_data, data_list = bidirectional_dataset_get(True, False, isTraining), bidirectional_dataset_get(
        False, False, isTraining), {}, []

    def word_status(status: dict):
        true_tfidf, false_tfidf = status["true"], status["false"]

        if true_tfidf > false_tfidf:
            return 1
        else:
            return 0

    def combined_and_separate(true_dict: dict, false_dict: dict):
        for key in true_dict:
            if key not in full_data:
                full_data[key] = {"true": 0, "false": 0, "label": None}
                full_data[key]["true"] = true_dict[key]
            else:
                full_data[key]["true"] += true_dict[key]

        for key in false_dict:
            if key not in full_data:
                full_data[key] = {"true": 0, "false": 0, "label": None}
                full_data[key]["false"] = false_dict[key]
            else:
                full_data[key]["false"] += false_dict[key]

        for key in full_data:
            full_data[key]["label"] = word_status(full_data[key])

        for key in full_data:
            encoder = {
                'word': key,
                'true': full_data[key]["true"],
                'false': full_data[key]["false"],
                'label': full_data[key]["label"]
            }
            data_list.append(encoder)

    combined_and_separate(data_true, data_false)

    if isTraining:
        extension = "_training"
    else:
        extension = "_testing"

    file = open(f"{path}raw/bidirectional{extension}.json", "w")
    json.dump(data_list, file)
    file.close()


def dataset_construction(isTraining: bool):
    bidirectional_dataset(isTraining)

    if isTraining:
        extension = "_training"
    else:
        extension = "_testing"

    rumors_df = pd.read_json(f"{path}raw/bidirectional{extension}.json")
    return rumors_df


def build_data_distribution(df: pd.DataFrame):
    df.plot(x="true", y="false", kind="scatter")
    plt.show()


# Execution Section
training_dataset = dataset_construction(True)
testing_dataset = dataset_construction(False)
# build_data_distribution(dataset)
