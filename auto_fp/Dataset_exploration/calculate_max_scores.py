import glob
import csv
import numpy as np

from utils import BASE_OUTPUT_DIR

dataset_list = [
    "ada", "austrilian", "blood", "christine",
    "Click_prediction_small", "credit", "EEG", "electricity",
    "emotion", "fibert", "forex", "gesture", "heart", "helena", "higgs",
    "house_data", "jannis", "jasmine", "kc1", "madeline", "numerai28.6",
    "pd_speech_features", "philippine", "phoneme", "thyroid-allhyper",
    "vehicle", "volkert", "wine_quality", "analcatdata_authorship",
    "gas-drift", "har", "hill", "ionosphere", "isolet", "mobile_price",
    "mozilla4", "nasa", "page", "robot", "run_or_walk", "spambase",
    "sylvine","wall-robot", "wilt", "covtype", "BNG_australian"
]
classifier_list = ["LR", "XGB", "MLP"]

headers = ["Dataset", "Classifier", "Score_with_FP"]
out_file = open(f'{BASE_OUTPUT_DIR}/Exp_dataset_exploration/results/max_scores_with_FP_vary_scenario.csv', 'w', newline ='')
with out_file:
    writer = csv.DictWriter(out_file, fieldnames = headers)
    writer.writeheader()
    for dataset in dataset_list:
        for classifier in classifier_list:
            file_list = glob.glob(f'{BASE_OUTPUT_DIR}/Exp_dataset_exploration/results/{dataset}_{classifier}*_score.csv')
            max_scores = []
            for file in file_list:
                max_score = 0
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if float(line) > max_score:
                            max_score = float(line)
                max_scores.append(max_score)
            print(f"{dataset},{classifier}")
            avg_max_score = np.mean(np.array(max_scores))
        
            temp_res = {}
            temp_res["Dataset"] = dataset
            temp_res["Classifier"] = classifier
            temp_res["Score_with_FP"] = avg_max_score
            writer.writerow(temp_res)