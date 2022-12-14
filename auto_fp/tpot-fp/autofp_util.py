import numpy as np
import csv

def load_data(dataset):
    train_dir = "~/Auto-FP/all_datasets_with_test/" + dataset + "_train.csv"
    valid_dir = "~/Auto-FP/all_datasets_with_test/" + dataset + "_valid.csv"
    test_dir = "~/Auto-FP/all_datasets_with_test/" + dataset + "_test.csv"

    X_train = []
    X_valid = []
    X_test = []
    y_train = []
    y_valid = []
    y_test =[]

    with open(train_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        line = -1
        for row in csv_reader:
            line += 1
            if (line == 0):
                continue
            if (not '?' in row[0: len(row) - 1]):
                X_train.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_train.append(int(float(row[-1])))

    with open(valid_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        line = -1
        for row in csv_reader:
            line += 1
            if (line == 0):
                continue
            if (not '?' in row[0: len(row) - 1]):
                X_valid.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_valid.append(int(float(row[-1])))

    with open(test_dir) as csvfile:
        csv_reader = csv.reader(csvfile)
        line = -1
        for row in csv_reader:
            line += 1
            if (line == 0):
                continue
            if (not '?' in row[0: len(row) - 1]):
                X_test.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                y_test.append(int(float(row[-1])))
    return np.array(X_train), np.array(X_valid), np.array(X_test), np.array(y_train), np.array(y_valid), np.array(y_test)
