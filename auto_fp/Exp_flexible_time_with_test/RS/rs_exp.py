import time
import sys
import random
import numpy as np
import argparse
import warnings

from sklearn.metrics import accuracy_score

from utils import DATA_DIR, BASE_OUTPUT_DIR, MAX_PIPE_LEN, operator_names
from utils import set_env, load_data, get_pipe, get_model, make_output_dir 

warnings.filterwarnings("ignore")

def run_rs(dataset, classifier, max_time_limit, max_len, seed):
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='RS')

    random.seed(seed)

    X_train, X_valid, y_train, y_valid = load_data(dataset)
    global_start = time.time()

    time_limit_reached = False
    while True:
        pick_start = time.time()
        pipe_length = random.randint(1, max_len)
        temp_pipe = []
        for j in range(pipe_length):
            temp_pipe.append(random.choice(operator_names))
        pick_end = time.time()
    
        generate_pipe_start = time.time()
        prep_pipe, prep_pipe_str = get_pipe(temp_pipe)
        generate_pipe_end = time.time()

        model = get_model(classifier)

        prep_train_start, prep_train_end = 0, 0
        prep_valid_start, prep_valid_end = 0, 0
        train_start, train_end = 0, 0
        predict_start, predict_end = 0, 0
        eval_score_start, eval_score_end = 0, 0
        try:
            prep_train_start = time.time()
            X_train_new = prep_pipe.fit_transform(X_train)
            prep_train_end = time.time()

            prep_valid_start = time.time()
            X_valid_new = prep_pipe.transform(X_valid)
            prep_valid_end = time.time()

            train_start = time.time()
            model.fit(np.array(X_train_new), np.array(y_train))
            train_end = time.time()
        
            predict_start = time.time()
            y_pred = model.predict(np.array(X_valid_new))
            predict_end = time.time()

            eval_score_start = time.time()
            score = accuracy_score(y_valid, y_pred)
            eval_score_end = time.time()
        except:
            score = 0
    
        global_mid = time.time()
        if (global_mid - global_start) >= max_time_limit:
            time_limit_reached = True 
        f = open(f'{output_dir}/rs_pick_time_{seed}.csv', 'a')
        f.write(str(pick_end - pick_start) + "\n")
        f = open(f'{output_dir}/rs_wallock_{seed}.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'{output_dir}/rs_pipe_{seed}.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{output_dir}/rs_score_{seed}.csv', 'a')
        f.write(str(score) + "\n")
        f = open(f'{output_dir}/rs_eval_time_{seed}.csv', 'a')
        f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                str(prep_train_end - prep_train_start) + "," +
                str(prep_valid_end - prep_valid_start) + "," +
                str(train_end - train_start) + "," +
                str(predict_end - predict_start) + "," +
                str(eval_score_end - eval_score_start) + "\n")
        if score != 0:
            if time_limit_reached:
                sys.exit()
        else:
            if time_limit_reached:
                sys.exit()
            else:
                continue

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,help="name of dataset")
    ap.add_argument("-c", "--classifier", required=True,help="name of classifier")
    ap.add_argument("-max_time", "--max_time_limit", required=True,help="number of max_time_limit")
    ap.add_argument("-seed", "--random_seed", required=True,help="random seed")
    ap.add_argument("-thread", "--max_running_thread", required=False, default=1, help="number of maximum running thread")
    args = vars(ap.parse_args())

    dataset = args['dataset']
    classifier = args['classifier']
    max_time_limit = int(args['max_time_limit'])
    seed = int(args["random_seed"])
    nthread = int(args["max_running_thread"])

    set_env(nthread=nthread)

    max_len = MAX_PIPE_LEN

    run_rs(dataset, classifier, max_time_limit, max_len, seed)
    
    
