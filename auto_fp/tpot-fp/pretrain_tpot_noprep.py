from tpot import TPOTClassifier
from sklearn.model_selection import PredefinedSplit
import numpy as np

import os
import csv
from joblib import Parallel, delayed

import autofp_util


def tpot_noprep_job(classifier, pretrain_mins, round_th, seed, dataset):
    simple_config_dict = {}
    if classifier == 'lr':
        simple_config_dict['sklearn.linear_model.LogisticRegression'] = {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False],
            'n_jobs': [1],
        }
    elif classifier == 'xgb':
        simple_config_dict['xgboost.XGBClassifier'] = {
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
            'n_jobs': [1],
            'verbosity': [0]
        }
    elif classifier == 'mlp':
        simple_config_dict['sklearn.neural_network.MLPClassifier'] = {
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
        }

    X_train, X_valid, X_test, y_train, y_valid, y_test = autofp_util.load_data(dataset)
    X_train_valid = np.concatenate((X_train, X_valid), axis=0)
    y_train_valid = np.concatenate((y_train, y_valid), axis=0)

    test_fold = np.concatenate((np.repeat(-1, X_train.shape[0]), np.repeat(0, X_valid.shape[0])),axis=0)
    ps = PredefinedSplit(test_fold)

    tpot = TPOTClassifier(config_dict=simple_config_dict, cv=ps, generations=None, max_eval_time_mins=int(pretrain_mins), max_time_mins=int(pretrain_mins),
            n_jobs=1, population_size=20, random_state=int(seed), verbosity=1)
    tpot.fit(X_train_valid, y_train_valid)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_' + classifier + '_noprep_' + pretrain_mins + 'min/round_' + round_th + '/seed_' + seed + '/tpot_' + dataset + '_pipeline.py')

if __name__ == "__main__":
    classifiers = ["lr", "mlp", "xgb"]
    pretrain_time = ["10"]
    rounds = ["0", "1", "2", "3", "4"]
    seeds = ["0", "42", "167", "578", "1440"]
    datasets = {
        "ada":[], "austrilian":[], "blood":[], "christine":[], "Click_prediction_small":[], "credit":[], "EEG":[],
        "electricity":[], "emotion":[], "fibert":[], "forex":[], "gesture":[], "heart":[], "helena":[], "higgs":[],
        "house_data":[], "jannis":[], "jasmine":[], "kc1":[], "madeline":[], "numerai28.6":[], "pd_speech_features":[],
        "philippine":[], "phoneme":[], "thyroid-allhyper":[], "vehicle":[], "volkert":[],
        "wine_quality":[], "analcatdata_authorship":[], "gas-drift":[], "har":[], "hill":[], "ionosphere":[],
        "isolet":[], "mobile_price":[], "mozilla4":[], "nasa":[], "page":[], "robot":[], "run_or_walk":[],
        "spambase":[], "sylvine":[], "wall-robot":[], "wilt":[], "covtype":[],
    }
    result = Parallel(n_jobs=75)(delayed(tpot_noprep_job)(classifier, pretrain_mins, round_th, seed, dataset) \
             for classifier in classifiers for dataset in datasets for pretrain_mins in pretrain_time \
             for round_th in rounds for seed in seeds)
    cur = 0
    for classifier in classifiers:
        print(classifier)
        for dataset in datasets:
            for round_th in rounds:
                for seed in seeds:
                    datasets[dataset].append(result[cur])
                    cur += 1
            print(dataset + " : " + str(sum(datasets[dataset])/25))
            datasets[dataset] = []
