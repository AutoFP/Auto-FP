import warnings
import numpy as np
import csv

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils import ORIGIN_DATA_DIR, DATA_DIR, load_data

warnings.filterwarnings("ignore")

dataset_list = [
    "ada",
    "austrilian",
    "blood",
    "BNG_austrilian",
    "christine",
    "Click_prediction_small",
    "covtype",
    "credit_1",
    "EEG",
    "electricity",
    "emotion",
    "fibert",
    "forex",
    "gesture",
    "heart",
    "helena",
    "higgs",
    "house_data",
    "jannis",
    "jasmine",
    "kc1",
    "madeline",
    "numerai28.6",
    "pd_speech_features",
    "philippine",
    "phoneme",
    "thyroid-allhyper",
    "vehicle",
    "volkert",
    "wine_quality",
    "analcatdata_authorship",
    "gas-drift",
    "har",
    "hill",
    "ionosphere",
    "isolet",
    "mobile_price",
    "mozilla4",
    "nasa",
    "page",
    "robot",
    "run_or_walk",
    "spambase",
    "sylvine",
    "wall-robot",
    "wilt"]

# Split dataset
def load_origin_data(dataset):
    data_dir = ORIGIN_DATA_DIR + dataset + ".csv"
    data = []
    label = []
    valid_percent = 0.2
    test_percent = 0.2
    with open(data_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        line = -1
        for row in csv_reader:
            line += 1
            if (not '?' in row[0: len(row) - 1]):
                data.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                label.append(float(row[-1]))
        X_pre_train, X_test, y_pre_train, y_test = train_test_split(data, label, test_size=test_percent, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_pre_train, y_pre_train, test_size=valid_percent, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def write_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    write_X_train = []
    write_X_valid = []
    write_X_test = []
    for i in range(len(X_train)):
        write_X_train.append([])
        for j in range(len(X_train[i])):
            write_X_train[i].append(X_train[i][j])
        write_X_train[i].append(y_train[i])
    for i in range(len(X_valid)):
        write_X_valid.append([])
        for j in range(len(X_valid[i])):
            write_X_valid[i].append(X_valid[i][j])
        write_X_valid[i].append(y_valid[i])
    for i in range(len(X_test)):
        write_X_test.append([])
        for j in range(len(X_test[i])):
            write_X_test[i].append(X_test[i][j])
        write_X_test[i].append(y_test[i])
    np.savetxt(DATA_DIR + dataset + "_train.csv", write_X_train, delimiter=",")
    np.savetxt(DATA_DIR + dataset + "_valid.csv", write_X_valid, delimiter=",")
    np.savetxt(DATA_DIR + dataset + "_test.csv", write_X_test, delimiter=",")

def get_train_and_valid_score(dataset):
    
    X_train_from_ori, X_valid_from_ori, X_test_from_ori, y_train_from_ori, y_valid_from_ori, y_test_from_ori = load_origin_data(dataset)
    write_data(X_train_from_ori, X_valid_from_ori, X_test_from_ori, y_train_from_ori, y_valid_from_ori, y_test_from_ori)
    X_train, X_valid, y_train, y_valid = load_data(dataset)
    
    print("Dataset Name:" + dataset)
    clf1 = LogisticRegression(random_state=0)
    clf1.fit(np.array(X_train), np.array(y_train))
    pred_train1 = clf1.predict(np.array(X_train))
    lr_train_score = accuracy_score(y_train, pred_train1)
    print("LR train score:" + str(lr_train_score))
    pred_valid1 = clf1.predict(np.array(X_valid))
    lr_valid_score = accuracy_score(y_valid, pred_valid1)
    print("LR valid score:" + str(lr_valid_score))

    clf2 = XGBClassifier(random_state=0)
    clf2.fit(np.array(X_train), np.array(y_train))
    pred_train2 = clf2.predict(np.array(X_train))
    xgb_train_score = accuracy_score(y_train, pred_train2)
    print("XGB train score:" + str(xgb_train_score))
    pred_valid2 = clf2.predict(np.array(X_valid))
    xgb_valid_score = accuracy_score(y_valid, pred_valid2)
    print("XGB valid score:" + str(xgb_valid_score))

    clf3 = MLPClassifier(random_state=0)
    clf3.fit(np.array(X_train), np.array(y_train))
    pred_train3 = clf3.predict(np.array(X_train))
    mlp_train_score = accuracy_score(y_train, pred_train3)
    print("MLP train score:" + str(mlp_train_score))
    pred_valid3 = clf3.predict(np.array(X_valid))
    mlp_valid_score = accuracy_score(y_valid, pred_valid3)
    print("MLP valid score:" + str(mlp_valid_score))

    return lr_train_score, lr_valid_score, xgb_train_score, xgb_valid_score, mlp_train_score, mlp_valid_score

output_path = f"{DATA_DIR}/Train_valid_score_no_fp.csv"
with open(output_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset","LR train score","LR valid score","XGB train score","XGB valid score","MLP train score","MLP valid score"])

for dataset in dataset_list:
    try:
        lr_train, lr_valid, xgb_train, xgb_valid, mlp_train, mlp_valid = get_train_and_valid_score(dataset)
        with open(output_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([str(dataset),str(lr_train),str(lr_valid),
                             str(xgb_train),str(xgb_valid),str(mlp_train),str(mlp_valid)])
    except:
        continue
    