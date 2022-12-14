import csv
import os

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, \
                                  MaxAbsScaler, MinMaxScaler, Binarizer, \
                                  PowerTransformer, QuantileTransformer

ORIGIN_DATA_DIR = "ORIGIN_DATA_DIR"
DATA_DIR = "DATA_DIR"
TRAIN_SUFFIX = "_train.csv"
VALID_SUFFIX = "_valid.csv"
TEST_SUFFIX = "_test.csv"

BASE_OUTPUT_DIR = "BASE_OUTPUT_DIR"

MAX_PIPE_LEN = 7

preprocessors_dic = {"binarizer": Binarizer,
                     "standardizer": StandardScaler,
                     "normalizer": Normalizer,
                     "maxabs": MaxAbsScaler,
                     "minmax": MinMaxScaler,
                     "power_trans": PowerTransformer,
                     "quantile_trans": QuantileTransformer}

operator_names = ["binarizer", "standardizer", "normalizer", "maxabs", "minmax", "power_trans", "quantile_trans"]

def load_data(dataset):
    train_data_dir = DATA_DIR + dataset + TRAIN_SUFFIX
    X_train = []
    y_train = []
    with open(train_data_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if (not '?' in row[0: len(row) - 1]):
                X_train.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_train.append(float(row[-1]))
                
    valid_data_dir = DATA_DIR + dataset + VALID_SUFFIX
    X_valid = []
    y_valid = []
    with open(valid_data_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if (not '?' in row[0: len(row) - 1]):
                X_valid.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_valid.append(float(row[-1]))
    return X_train, X_valid, y_train, y_valid

def get_pipe(PARAMS):
    pipe_length = len(PARAMS)
    pipe_str = ""
    if (pipe_length == 1):
        op_name = PARAMS[0]
        pipe_str = str(op_name)
        return preprocessors_dic.get(str(op_name)), pipe_str
    op1_name = PARAMS[0]
    op2_name = PARAMS[1]
    if op1_name == "quantile_trans":
        op1 = make_pipeline(pipe, preprocessors_dic.get(str(op_name))(random_state=0))
    else:
        op1 = make_pipeline(pipe, preprocessors_dic.get(str(op_name))())
    if op2_name == "quantile_trans":
        op2 = make_pipeline(pipe, preprocessors_dic.get(str(op_name))(random_state=0))
    else:
        op2 = make_pipeline(pipe, preprocessors_dic.get(str(op_name))())
    pipe = make_pipeline(op1, op2)
    pipe_str = op1_name + "," + op2_name

    for i in range(2, pipe_length):
        op_name = PARAMS[i]
        pipe_str += "," + op_name
        if op_name == "quantile_trans":
            pipe = make_pipeline(pipe, preprocessors_dic.get(str(op_name))(random_state=0))
        else:
            pipe = make_pipeline(pipe, preprocessors_dic.get(str(op_name))())
    return pipe, pipe_str

def get_model(classifier):
    if classifier == 'LR':
        model = LogisticRegression(random_state=0, n_jobs=1)
    elif classifier == 'XGB': 
        model =  XGBClassifier(random_state=0, nthread=1, n_jobs=1)
    elif classifier == 'MLP':
        model = MLPClassifier(random_state=0)
    return model

def set_env(nthread = 1):
    os.environ["OMP_NUM_THREADS"] = str(nthread)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthread)
    os.environ["MKL_NUM_THREADS"] = str(nthread)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthread)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthread)

def make_output_dir(dataset, classifier, max_time_limit, algorithm):

    dataset_path = f"{BASE_OUTPUT_DIR}/{dataset}"
    time_limit_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}"
    classifier_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}/{classifier}"
    algorithm_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}/{classifier}/{algorithm}"
    

    isDatasetPathExist = os.path.exists(dataset_path)
    if not isDatasetPathExist:
        os.makedirs(dataset_path)

    isTimePathExist = os.path.exists(time_limit_path)
    if not isTimePathExist:
        os.makedirs(time_limit_path)

    isClassifierPathExist = os.path.exists(classifier_path)
    if not isClassifierPathExist:
        os.makedirs(classifier_path)

    isAlgPathExist = os.path.exists(algorithm_path)
    if not isAlgPathExist:
        os.makedirs(algorithm_path)

    return algorithm_path

def make_output_dir_extended_space(dataset, classifier, max_time_limit, algorithm, type=("balanced", "one-step")):

    if type == ("balanced", "one-step"):
        dataset_path = f"{BASE_OUTPUT_DIR}/{dataset}"
        time_limit_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space"
        classifier_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space/{classifier}"
        algorithm_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space/{classifier}/{algorithm}"
    
    if type == ("balanced", "two-step"):
        dataset_path = f"{BASE_OUTPUT_DIR}/{dataset}"
        time_limit_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_with_heuristic"
        classifier_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_with_heuristic/{classifier}"
        algorithm_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_with_heuristic/{classifier}/{algorithm}"

    if type == ("imbalanced", "one-step"):
        dataset_path = f"{BASE_OUTPUT_DIR}/{dataset}"
        time_limit_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_imbalanced"
        classifier_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_imbalanced/{classifier}"
        algorithm_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_imbalanced/{classifier}/{algorithm}"

    if type == ("imbalanced", "two-step"):
        dataset_path = f"{BASE_OUTPUT_DIR}/{dataset}"
        time_limit_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_imbalanced_with_heuristic"
        classifier_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_imbalanced_with_heuristic/{classifier}"
        algorithm_path = f"{BASE_OUTPUT_DIR}/{dataset}/max_time{max_time_limit}_extended_space_imbalanced_with_heuristic/{classifier}/{algorithm}"
    

    isDatasetPathExist = os.path.exists(dataset_path)
    if not isDatasetPathExist:
        os.makedirs(dataset_path)

    isTimePathExist = os.path.exists(time_limit_path)
    if not isTimePathExist:
        os.makedirs(time_limit_path)

    isClassifierPathExist = os.path.exists(classifier_path)
    if not isClassifierPathExist:
        os.makedirs(classifier_path)

    isAlgPathExist = os.path.exists(algorithm_path)
    if not isAlgPathExist:
        os.makedirs(algorithm_path)

    return algorithm_path