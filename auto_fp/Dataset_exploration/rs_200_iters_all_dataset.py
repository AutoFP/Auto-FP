import warnings
import numpy as np
import time

from sklearn.metrics import accuracy_score

from utils import BASE_OUTPUT_DIR, operator_names
from utils import load_data, get_pipe, get_model

warnings.filterwarnings("ignore")

dataset_list = [
    "ada",
    "austrilian",
    "blood",
    "christine",
    "Click_prediction_small",
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
    "wilt",
    "covtype"]

classifier_list = ["LR", "XGB", "MLP"]
all_list = [i for i in range(0, 10000)]

def run_200_rs(dataset, classifier, seed):
    max_iterations = 200
    current_iter = 0
    max_score = 0
    max_len = 7
    
    np.random.seed(seed)
    
    X_train, X_valid, y_train, y_valid = load_data(dataset)
        
    while (current_iter < max_iterations):
        rand_start = time.time()
        length = np.random.randint(1, max_len)
        temp_pipe = []
        for i in range(length):
            temp_pipe.append(operator_names[np.random.randint(1, len(operator_names)) - 1])
        rand_end = time.time()
        pick_time = rand_end - rand_start

        generate_pipe_start = time.time()
        prep_pipe, prep_pipe_str = get_pipe(temp_pipe)
        generate_pipe_end = time.time()

        model = get_model(classifier)

        try:
            X_train_new = prep_pipe.fit_transform(X_train)
            X_valid_new = prep_pipe.transform(X_valid)
        
            model.fit(np.array(X_train_new), np.array(y_train))
        
            y_pred = model.predict(np.array(X_valid_new))
            score = accuracy_score(y_valid, y_pred)
        except:
            score = 0
        current_iter += 1
        if score > max_score:
            max_score = score

        f = open(f'{BASE_OUTPUT_DIR}/Exp_dataset_exploration/results/{dataset}_{classifier}_seed_{seed}_pipe.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{BASE_OUTPUT_DIR}/Exp_dataset_exploration/results/{dataset}_{classifier}_seed_{seed}_score.csv', 'a')
        f.write(str(score) + "\n")
    print(max_score)
    return max_score

if __name__ == "__main__":
    f = open(f'{BASE_OUTPUT_DIR}/Exp_dataset_exploration/results/max_score_record_200_iters.csv', 'w')
    f.write('dataset,classifier,avg_max_score' + '\n')
    f.close()
    for dataset in dataset_list:
        for classifier in classifier_list:
            avg_max_score = 0
            try:
                max_score_list = []
                for i in range(5):
                    seed = np.random.choice(all_list)
                    temp_max_score = run_200_rs(dataset, classifier, seed)
                    max_score_list.append(temp_max_score)
                avg_max_score = np.mean(np.array(max_score_list))
            except:
                continue
            f = open(f'{BASE_OUTPUT_DIR}/Exp_dataset_exploration/results/max_score_record_200_iters.csv', 'a')
            f.write(dataset + ',' + classifier + ',' + str(avg_max_score) + '\n')
            f.close()
