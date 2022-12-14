from tpot import TPOTClassifier
from sklearn.model_selection import PredefinedSplit
import numpy as np

import os
import csv
from joblib import Parallel, delayed

import autofp_util

def tpot_default_job(classifier, pretrain_mins, round_th, seed, dataset):
    simple_config_dict = {

        # Preprocesssors
        'sklearn.preprocessing.Binarizer': {
        #    'threshold': np.arange(0.0, 1.01, 0.05)
        },

        #'sklearn.decomposition.FastICA': {
        #    'tol': np.arange(0.0, 1.01, 0.05)
        #},

        #'sklearn.cluster.FeatureAgglomeration': {
        #    'linkage': ['ward', 'complete', 'average'],
        #    'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        #},

        'sklearn.preprocessing.MaxAbsScaler': {},

        'sklearn.preprocessing.MinMaxScaler': {},

        'sklearn.preprocessing.Normalizer': {
        #    'norm': ['l1', 'l2', 'max']
        },

        #'sklearn.kernel_approximation.Nystroem': {
        #    'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        #    'gamma': np.arange(0.0, 1.01, 0.05),
        #    'n_components': range(1, 11)
        #},

        #'sklearn.decomposition.PCA': {
        #    'svd_solver': ['randomized'],
        #    'iterated_power': range(1, 11)
        #},

        #'sklearn.preprocessing.PolynomialFeatures': {
        #    'degree': [2],
        #    'include_bias': [False],
        #    'interaction_only': [False]
        #},

        #'sklearn.kernel_approximation.RBFSampler': {
        #    'gamma': np.arange(0.0, 1.01, 0.05)
        #},

        #'sklearn.preprocessing.RobustScaler': {},

        'sklearn.preprocessing.StandardScaler': {},

        #'tpot.builtins.ZeroCount': {},

        #'tpot.builtins.OneHotEncoder': {
        #    'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        #    'sparse': [False],
        #    'threshold': [10]
        #},

        # Selectors
        #'sklearn.feature_selection.SelectFwe': {
        #    'alpha': np.arange(0, 0.05, 0.001),
        #    'score_func': {
        #        'sklearn.feature_selection.f_classif': None
        #    }
        #},

        #'sklearn.feature_selection.SelectPercentile': {
        #    'percentile': range(1, 100),
        #    'score_func': {
        #        'sklearn.feature_selection.f_classif': None
        #    }
        #},

        #'sklearn.feature_selection.VarianceThreshold': {
        #    'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        #},

        #'sklearn.feature_selection.RFE': {
        #    'step': np.arange(0.05, 1.01, 0.05),
        #    'estimator': {
        #        'sklearn.ensemble.ExtraTreesClassifier': {
        #            'n_estimators': [100],
        #            'criterion': ['gini', 'entropy'],
        #            'max_features': np.arange(0.05, 1.01, 0.05)
        #        }
        #    }
        #},

        #'sklearn.feature_selection.SelectFromModel': {
        #    'threshold': np.arange(0, 1.01, 0.05),
        #    'estimator': {
        #        'sklearn.ensemble.ExtraTreesClassifier': {
        #            'n_estimators': [100],
        #            'criterion': ['gini', 'entropy'],
        #            'max_features': np.arange(0.05, 1.01, 0.05)
        #        }
        #    }
        #}

    }

    if classifier == 'lr':
        simple_config_dict['sklearn.linear_model.LogisticRegression'] = {
            'n_jobs': [1]
        }
    elif classifier == 'xgb':
        simple_config_dict['xgboost.XGBClassifier'] = {
            'n_jobs': [1]
        }
    elif classifier == 'mlp':
        simple_config_dict['sklearn.neural_network.MLPClassifier'] = {}

    X_train, X_valid, X_test, y_train, y_valid, y_test = autofp_util.load_data(dataset)
    X_train_valid = np.concatenate((X_train, X_valid), axis=0)
    y_train_valid = np.concatenate((y_train, y_valid), axis=0)
    test_fold = np.concatenate((np.repeat(-1, X_train.shape[0]), np.repeat(0, X_valid.shape[0])),axis=0)
    ps = PredefinedSplit(test_fold)

    tpot = TPOTClassifier(config_dict=simple_config_dict, cv=ps, generations=None, max_eval_time_mins=int(pretrain_mins), max_time_mins=int(pretrain_mins),
            n_jobs=1, population_size=20, random_state=int(seed), verbosity=1)
    tpot.fit(X_train_valid, y_train_valid)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_' + classifier + '_default_' + pretrain_mins + 'min/round_' + round_th + '/seed_' + seed + '/tpot_' + dataset + '_pipeline.py')

if __name__ == "__main__":
    classifiers = ["lr", "mlp", "xgb"]
    pretrain_time = ["10"]
    rounds = ["0", "1", "2", "3", "4"]
    seeds = ["0", "42", "167", "578", "1440"]
    datasets = {
        "ada":[], "austrilian":[], "blood":[], "christine":[], "Click_prediction_small":[], "credit":[],
        "EEG":[], "electricity":[], "emotion":[], "fibert":[], "forex":[], "gesture":[], "heart":[],
        "helena":[], "higgs":[], "house_data":[], "jannis":[], "jasmine":[], "kc1":[], "madeline":[],
        "numerai28.6":[], "pd_speech_features":[], "philippine":[], "phoneme":[], "thyroid-allhyper":[],
        "vehicle":[], "volkert":[], "wine_quality":[], "analcatdata_authorship":[], "gas-drift":[], "har":[],
        "hill":[], "ionosphere":[], "isolet":[], "mobile_price":[], "mozilla4":[], "nasa":[], "page":[],
        "robot":[], "run_or_walk":[], "spambase":[], "sylvine":[], "wall-robot":[], "wilt":[], "covtype":[],
    }

    result = Parallel(n_jobs=75)(delayed(tpot_default_job)(classifier, pretrain_mins, round_th, seed, dataset) \
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
