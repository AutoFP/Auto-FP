import os
import re

classifiers = [
    "./tpot_lr_ps_step_10min",
    "./tpot_mlp_ps_step_10min",
    "./tpot_xgb_ps_step_10min"
]

for top in classifiers:
    datasets={
        "ada":[], "austrilian":[], "blood":[], "christine":[], "Click_prediction_small":[], "credit":[], "EEG":[],
        "electricity":[], "emotion":[], "fibert":[], "forex":[], "gesture":[], "heart":[], "helena":[], "higgs":[],
        "house_data":[], "jannis":[], "jasmine":[], "kc1":[], "madeline":[], "numerai28.6":[], "pd_speech_features":[],
        "philippine":[], "phoneme":[], "thyroid-allhyper":[], "vehicle":[], "volkert":[], "wine_quality":[],
        "analcatdata_authorship":[], "gas-drift":[], "har":[], "hill":[], "ionosphere":[], "isolet":[], "mobile_price":[],
        "mozilla4":[], "nasa":[], "page":[], "robot":[], "run_or_walk":[], "spambase":[], "sylvine":[], "wall-robot":[],
        "wilt":[], "covtype":[],
    }
    for root, dirs, files in os.walk(top):
        for name in files:
            file_name = os.path.join(root, name)
            for dataset in datasets:
                if ("_" + dataset + "_") in file_name:
                    f = open(file_name, 'r')
                    for line in f.readlines():
                        if "score" in line:
                            datasets[dataset].append(float(re.search(r'-?\d+\.?\d*e?\+?\d*', line).group(0)))
                            break;
    print(top)
    for dataset, score in datasets.items():
        print(dataset + " : " +  (str(sum(score) / 25.0) if len(score) == 25 else ""))
