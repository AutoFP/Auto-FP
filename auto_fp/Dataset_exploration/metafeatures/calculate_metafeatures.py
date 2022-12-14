from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels
import numpy as np
import csv
import logging
logger = logging.getLogger("03_calculate_metafeatures")

from utils import ORIGIN_DATA_DIR

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
            #train_X.append(list(map(lambda x: float(x), row[2: len(row)])))
            if (not '?' in row[0: len(row) - 1]):
                data.append(list(map(lambda x: float(x), row[0: len(row) - 1])))
                label.append(float(row[-1]))
    return data, label

dataset_list = [
    "ada", "austrilian", "blood", "christine",
    "Click_prediction_small", "credit", "EEG", "electricity",
    "emotion", "fibert", "forex", "gesture", "heart", "helena", "higgs",
    "house_data", "jannis", "jasmine", "kc1", "madeline", "numerai28.6",
    "pd_speech_features", "philippine", "phoneme", "thyroid-allhyper",
    "vehicle", "volkert", "wine_quality", "analcatdata_authorship",
    "gas-drift", "har", "hill", "ionosphere", "isolet", "mobile_price",
    "mozilla4", "nasa", "page", "robot", "run_or_walk", "spambase",
    "sylvine", "wall-robot", "wilt", "covtype"]
    
keys = ['Dataset','PCASkewnessFirstPC', 'PCAKurtosisFirstPC', 'PCAFractionOfComponentsFor95PercentVariance',
        'Landmark1NN', 'LandmarkRandomNodeLearner', 'LandmarkDecisionNodeLearner', 'LandmarkDecisionTree',
        'LandmarkNaiveBayes', 'LandmarkLDA', 'ClassEntropy', 'SkewnessSTD', 'SkewnessMean', 'SkewnessMax',
        'SkewnessMin', 'KurtosisSTD', 'KurtosisMean', 'KurtosisMax', 'KurtosisMin', 'SymbolsSum', 'SymbolsSTD',
        'SymbolsMean', 'SymbolsMax', 'SymbolsMin', 'ClassProbabilitySTD', 'ClassProbabilityMean',
        'ClassProbabilityMax', 'ClassProbabilityMin', 'InverseDatasetRatio', 'DatasetRatio',
        'NumberOfMissingValues', 'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues',
        'NumberOfFeatures', 'NumberOfClasses', 'NumberOfInstances', 'LogInverseDatasetRatio', 'LogDatasetRatio',
        'PercentageOfMissingValues', 'PercentageOfFeaturesWithMissingValues', 'PercentageOfInstancesWithMissingValues',
        'LogNumberOfFeatures', 'LogNumberOfInstances']
file = open('results/metafeatures.csv', 'w', newline ='')
with file:
    writer = csv.DictWriter(file, fieldnames = keys)
    writer.writeheader()

    for dataset in dataset_list:
        data, label = load_origin_data(dataset)
        data = np.array(data)
        label = np.array(label)
        mf = calculate_all_metafeatures_with_labels(data, label, dataset_name=dataset, logger=logger)

        values = {'Dataset': dataset}
        for key in mf.keys():
            if key not in ('PCA', 'Skewnesses', 'Kurtosisses', 'NumSymbols', 'ClassOccurences', 'MissingValues'):
                values[str(key)] = float(mf[key].value)
        writer.writerow(values)
        print(dataset)

