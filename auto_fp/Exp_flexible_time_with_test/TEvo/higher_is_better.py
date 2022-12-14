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

def mutate(parent, mutate_way_seed, op_seed, pos_seed):
    mutations = []
    if (len(parent) == 1):
        mutations = ["add", "replace"]
    elif (len(parent) == max_len):
        mutations = ["delete", "replace", "switch"]
    else:
        mutations = ["add", "delete", "replace", "switch"]
    np.random.seed(mutate_way_seed)
    mutate_type = np.random.choice(mutations)

    child = []
    if (mutate_type == "add"):
        if len(parent) == 1:
            pos = 0
        else:
            np.random.seed(pos_seed)
            pos = np.random.randint(0, len(parent) - 1)
        np.random.seed(op_seed)
        op = np.random.choice(operator_names)
        for i in range(pos + 1):
            child.append(parent[i])
        child.append(op)
        for i in range(pos + 1, len(parent)):
            child.append(parent[i])
    elif (mutate_type == "delete"):
        if len(parent) == 1:
            pos = 0
        else:
            np.random.seed(pos_seed)
            pos = np.random.randint(0, len(parent) - 1)
        for i in range(pos):
            child.append(parent[i])
        for i in range(pos + 1, len(parent)):
            child.append(parent[i])
    elif (mutate_type == "replace"):
        if len(parent) == 1:
            pos = 0
        else:
            np.random.seed(pos_seed)
            pos = np.random.randint(0, len(parent) - 1)
        np.random.seed(op_seed)
        op = np.random.choice(operator_names)
        for i in range(pos):
            child.append(parent[i])
        child.append(op)
        for i in range(pos + 1, len(parent)):
            child.append(parent[i])
    elif (mutate_type == "switch"):
        np.random.seed(pos_seed)
        pos = np.random.choice(np.arange(len(parent)), size=2, replace=False)
        pos1 = min(pos)
        pos2 = max(pos)
        for i in range(pos1):
            child.append(parent[i])
        child.append(parent[pos2])
        for i in range(pos1 + 1, pos2):
            child.append(parent[i])
        child.append(parent[pos1])
        for i in range(pos2 + 1, len(parent)):
            child.append(parent[i])
    return child

def run_tevo_h(dataset, classifier, max_time_limit, max_len, seed):
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='TEVO_H')
    
    all_list = [i for i in range(0, 5000000)]
    np.random.seed(seed)
    pipe_seeds = np.random.choice(all_list, 1000000, replace=False)
    mutate_way_seeds = np.random.choice(all_list, 1000000, replace=False)
    op_seeds = np.random.choice(all_list, 1000000, replace=False)
    pos_seeds = np.random.choice(all_list, 1000000, replace=False)

    # Generate population
    P = 100
    S = 25
    print("P=" + str(P))
    print("S=" + str(S))
    Cycle_num = 1000000

    population = []
    population_score = []
    history = []
    history_score = []

    X_train, X_valid, y_train, y_valid = load_data(dataset)
    global_start = time.time()

    time_limit_reached = False

    while (len(population) < P):
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
        prep_train_start, prep_train_end = 0, 0
        prep_valid_start, prep_valid_end = 0, 0
        train_start, train_end = 0, 0
        pred_start, pred_end = 0, 0
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

            pred_start = time.time()
            y_pred = model.predict(np.array(X_valid_new))
            pred_end = time.time()

            eval_score_start = time.time()
            score = accuracy_score(y_valid, y_pred)
            eval_score_end = time.time()
        except:
            score = 0
        global_mid = time.time()
        if (global_mid - global_start) >= max_time_limit:
            time_limit_reached = True 
        f = open(f'{output_dir}/higher_pick_time_{seed}.csv', 'a')
        f.write(str(pick_time) + "\n")
        f = open(f'{output_dir}/higher_wallock_{seed}.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'{output_dir}/higher_pipe_{seed}.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{output_dir}/higher_score_{seed}.csv', 'a')
        f.write(str(score) + "\n")
        f = open(f'{output_dir}/higher_eval_time_{seed}.csv', 'a')
        f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                str(prep_train_end - prep_train_start) + "," +
                str(prep_valid_end - prep_valid_start) + "," +
                str(train_end - train_start) + "," +
                str(pred_end - pred_start) + "," +
                str(eval_score_end - eval_score_start) + "\n")
    
        if score != 0:
            if time_limit_reached:
                sys.exit()
        else:
            if time_limit_reached:
                sys.exit()
            else:
                continue

        population.append(temp_pipe)
        population_score.append(score)
        history.append(temp_pipe)
        history_score.append(score)

    # Run cycles
    update_start = None
    update_end = None
    count = -1
    while (len(history) < Cycle_num):
        count +=1 
        # Select sample
        sample_start = time.time()
        sample = []
        sample_score = []
        idx = np.arange(0, P)
        np.random.seed(pipe_seeds[count])
        selected_idx = np.random.choice(idx, S, replace=False)
       
        for item in selected_idx:
            sample.append(population[item])
            sample_score.append(population_score[item])
        sample_end = time.time()

        # Get parent
        mutate_start = time.time()
        parent = sample[np.argmax(sample_score)]
        child = mutate(parent, mutate_way_seed=mutate_way_seeds[count],
                       op_seed=op_seeds[count], pos_seed=pos_seeds[count])
        mutate_end = time.time()

        if update_start is None:
            pick_time = (sample_end - sample_start) + (mutate_end - mutate_start)
        else:
            pick_time = (update_end - update_start) + (sample_end - sample_start) + (mutate_end - mutate_start)

        generate_pipe_start = time.time()
        prep_pipe, prep_pipe_str = get_pipe(child)
        generate_pipe_end = time.time()

        model = get_model(classifier)
        prep_train_start, prep_train_end = 0, 0
        prep_valid_start, prep_valid_end = 0, 0
        train_start, train_end = 0, 0
        pred_start, pred_end = 0, 0
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

            pred_start = time.time()
            y_pred = model.predict(np.array(X_valid_new))
            pred_end = time.time()

            eval_score_start = time.time()
            child_score = accuracy_score(y_valid, y_pred)
            eval_score_end = time.time()
        except:
            child_score = 0
        global_mid = time.time()
        if (global_mid - global_start) >= max_time_limit:
            time_limit_reached = True 
        f = open(f'{output_dir}/TEVO_H/higher_pick_time_{seed}.csv', 'a')
        f.write(str(pick_time) + "\n")
        f = open(f'{output_dir}/TEVO_H/higher_wallock_{seed}.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'{output_dir}/higher_pipe_{seed}.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{output_dir}/higher_score_{seed}.csv', 'a')
        f.write(str(child_score) + "\n")
        f = open(f'{output_dir}/higher_eval_time_{seed}.csv', 'a')
        f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                str(prep_train_end - prep_train_start) + "," +
                str(prep_valid_end - prep_valid_start) + "," +
                str(train_end - train_start) + "," +
                str(pred_end - pred_start) + "," +
                str(eval_score_end - eval_score_start) + "\n")
        
        if child_score != 0:
            if time_limit_reached:
                sys.exit()
        else:
            if time_limit_reached:
                sys.exit()
            else:
                continue

        update_start = time.time()
        # Higher is better
        min_idx = selected_idx[np.argmin(sample_score)]
        population.pop(min_idx)
        population_score.pop(min_idx)
        population.append(child)
        population_score.append(child_score)
        history.append(child)
        history_score.append(child_score)
        update_end = time.time()

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

    run_tevo_h(dataset, classifier, max_time_limit, max_len, seed)
