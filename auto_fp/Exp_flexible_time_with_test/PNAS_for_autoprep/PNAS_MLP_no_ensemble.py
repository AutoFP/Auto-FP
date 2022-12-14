import time
import sys
import random
import numpy as np
import argparse
import warnings

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor

from utils import DATA_DIR, BASE_OUTPUT_DIR, MAX_PIPE_LEN, operator_names
from utils import set_env, load_data, get_pipe, get_model, make_output_dir 

warnings.filterwarnings("ignore")

def get_max_iterations(dataset, classifier, max_time_limit):
    result = 0
    if classifier == 'LR':
        if dataset == "austrilian":
            result = 120000
        if dataset == "blood":
            result = 160000
        if dataset == "emotion":
            result = 33000
        if dataset == "forex":
            result = 5000
        if dataset == "heart":
            result = 160000
        if dataset == "jasmine":
            result = 6000
        if dataset == "madeline":
            result = 3200
        if dataset == "pd_speech_features":
            result = 4300
        if dataset == "wine_quality":
            result = 9000
        if dataset == "thyroid-allhyper":
            result = 15000
    if classifier == 'XGB':
        if dataset == "austrilian":
            result = 55000
        if dataset == "blood":
            result = 110000
        if dataset == "emotion":
            result = 25000
        if dataset == "forex":
            result = 700
        if dataset == "heart":
            result = 75000
        if dataset == "jasmine":
            result = 2500
        if dataset == "madeline":
            result = 500
        if dataset == "pd_speech_features":
            result = 1300
        if dataset == "wine_quality":
            result = 900
        if dataset == "thyroid-allhyper":
            result = 2000
    if classifier == 'MLP':
        if dataset == "austrilian":
            result = 6000
        if dataset == "blood":
            result = 13000
        if dataset == "emotion":
            result = 5500
        if dataset == "forex":
            result = 400
        if dataset == "heart":
            result = 10000
        if dataset == "jasmine":
            result = 600
        if dataset == "madeline":
            result = 500
        if dataset == "pd_speech_features":
            result = 1000
        if dataset == "wine_quality":
            result = 750
        if dataset == "thyroid-allhyper":
            result = 1700
    result = int(result * (max_time_limit / float(3600)))
    return result
                 
def embedding(structure):
    return (np.sum(structure, axis=0) / (len(structure))).tolist()

def one_hot(structure):
    encode = np.zeros((len(structure), len(operator_names)))
    for i in range(len(structure)):
        encode[i][structure[i]] = 1
    return encode.tolist()

def run_exp(dataset, classifier, max_time_limit, max_len, seed, K_num = 50):
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='PNAS_MLP_NO_ENSEMBLE')

    # [0, 42, 167,578,1440]
    random.seed(seed)

    X_train, X_valid, y_train, y_valid = load_data(dataset)
    global_start = time.time()
    print("K_NUM = " + str(K_num))

    # History pipes
    history = []
    embedded_history = []
    validate_acc =[]
    pipelines = []

    # Predictor
    # [0, 42, 167,578,1440]
    predictor = MLPRegressor(random_state=seed)
  
    # Init predictor
    time_limit_reached = False
    for i in range(len(operator_names)):
        generate_pipe_start, pick_start = time.time(), time.time()
        prep_pipe, prep_pipe_str = get_pipe([operator_names[i]])
        generate_pipe_end, pick_end = time.time(), time.time()
        pick_time = pick_end - pick_start
        arch = [i]
        model = get_model(classifier)
        prep_train_start, prep_train_end = 0, 0
        prep_valid_start, prep_valid_end = 0, 0
        train_start, train_end = 0, 0
        pred_start, pred_end = 0, 0
        eval_score_start, eval_score_end = 0, 0

        pipelines.append(prep_pipe)
        history.append([i])
        try:
            prep_train_start =  time.time()
            if len(arch) > 0:
                X_train_new = prep_pipe.fit_transform(X_train)
            else:
                X_train_new = np.array(X_train)
            prep_train_end = time.time()

            prep_valid_start = time.time()
            if len(arch) > 0:
                X_valid_new = prep_pipe.transform(X_valid)
            else:
                X_valid_new = np.array(X_valid)
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

            validate_acc.append(accuracy_score(y_valid, y_pred))

            embedded_history.append(embedding(one_hot([i])))

        except:
            score = 0
        global_mid = time.time()
        if global_mid - global_start >= max_time_limit:
            time_limit_reached = True
        else:
            time_limit_reached = False
        f = open(f'{output_dir}/pnas_mlp_no_ensemble_pick_time_{seed}.csv', 'a')
        f.write(str(pick_time) + "\n")
        f = open(f'{output_dir}/pnas_mlp_no_ensemble_wallock_{seed}.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'{output_dir}/pnas_mlp_no_ensemble_pipe_{seed}.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{output_dir}/pnas_mlp_no_ensemble_score_{seed}.csv', 'a')
        f.write(str(score) + "\n")
        f = open(f'{output_dir}/pnas_mlp_no_ensemble_eval_time_{seed}.csv', 'a')
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

            #continue
    update_start = time.time()
    predictor.fit(embedded_history, validate_acc)
    update_end = time.time()

    # Start main exp
    K = K_num

    temp_pipes = [history.copy()]
    temp_acc = [validate_acc.copy()]

    for length in range(max_len - 1):
        pick_start = time.time()
        generate_pipes = []
        generate_encode =[]
        generate_acc = []
        for i in range(len(temp_pipes[-1])):
            pre_pipe = temp_pipes[-1][i].copy()
            for j in range(len(operator_names)):
                generate_pipes.append(pre_pipe + [j])
        for i in range(len(generate_pipes)):
            encode = embedding(one_hot(generate_pipes[i]))
            generate_encode.append(encode)
        generate_acc = predictor.predict(generate_encode)

        # Get top-k result
        topK_pipe = []
        topK_acc = []
        temp_topK_pipe = []

        if (K > len(generate_pipes)):
            temp_topK_pipe = generate_pipes.copy()
        else:
            ind = np.argpartition(generate_acc, -K)[-K:]
            for item in ind:
                temp_topK_pipe.append(generate_pipes[item])
        pick_end = time.time()
        if update_start is None:
            pick_time = pick_end - pick_start
        else:
            pick_time = (pick_end - pick_start) + (update_end - update_start)
        pick_time = pick_time / min(K, len(generate_pipes))

        # Train and evaluate top-k
        k_Value = min(K, len(generate_pipes))
        for i in range(k_Value):
            generate_pipe_start = time.time()
            arch = temp_topK_pipe[i]
            if (len(arch) == 1):
                prep_pipe, prep_pipe_str = get_pipe([operator_names[arch[0]]])
            elif len(arch) > 1:
                temp_pipe = []
                for ii in range(2, len(arch)):
                    temp_pipe.append(operator_names[arch[ii]])
                prep_pipe, prep_pipe_str = get_pipe(temp_pipe)
            generate_pipe_end = time.time()

            model = get_model(classifier)
            prep_train_start, prep_train_end = 0, 0
            prep_valid_start, prep_valid_end = 0, 0
            train_start, train_end = 0, 0
            pred_start, pred_end = 0, 0
            eval_score_start, eval_score_end = 0, 0
            try:
                prep_train_start =  time.time()
                if len(arch) > 0:
                    X_train_new = prep_pipe.fit_transform(X_train)
                else:
                    X_train_new = np.array(X_train)
                prep_train_end = time.time()

                prep_valid_start = time.time()
                if len(arch) > 0:
                    X_valid_new = prep_pipe.transform(X_valid)
                else:
                    X_valid_new = np.array(X_valid)
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

                topK_acc.append(score)
                pipelines.append(prep_pipe)
                topK_pipe.append(temp_topK_pipe[i])

            except:
                score = 0
            global_mid = time.time()
            time_limit_reached = True if global_mid - global_start >= max_time_limit else False
            f = open(f'{output_dir}/pnas_mlp_no_ensemble_pick_time_{seed}.csv', 'a')
            f.write(str(pick_time) + "\n")
            f = open(f'{output_dir}/pnas_mlp_no_ensemble_wallock_{seed}.csv', 'a')
            f.write(str(global_mid - global_start) + "\n")
            f = open(f'{output_dir}/pnas_mlp_no_ensemble_pipe_{seed}.csv', 'a')
            f.write(prep_pipe_str + "\n")
            f = open(f'{output_dir}/pnas_mlp_no_ensemble_score_{seed}.csv', 'a')
            f.write(str(score) + "\n")
            f = open(f'{output_dir}/pnas_mlp_no_ensemble_eval_time_{seed}.csv', 'a')
            f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                        str(prep_train_end - prep_train_start) + "," +
                        str(prep_valid_end - prep_valid_start) + "," +
                        str(train_end - train_start) + "," +
                        str(pred_end - pred_start) + "," +
                        str(eval_score_end - eval_score_start) + "\n")
                #continue
            if score != 0:
                if time_limit_reached:
                    sys.exit()
            else:
                if time_limit_reached:
                    sys.exit()
                else:
                    continue

        temp_pipes.append(topK_pipe)
        temp_acc.append(topK_acc)

        # Update predictor
        update_start = time.time()
        history += topK_pipe
        validate_acc += topK_acc
        for i in range(len(topK_pipe)):
            embedded_history.append(embedding(one_hot(topK_pipe[i])))
        predictor.fit(embedded_history, validate_acc)
        update_end = time.time()
        print(length)

    print("Max acc: " + str(max(validate_acc)))
    print("Total pipe num: " + str(len(validate_acc)))
    return max(validate_acc), len(validate_acc), validate_acc, history

def run_pnas_mlp_no_ensemble(dataset, classifier, max_time_limit, max_len, seed):
    # set different max_iter specially for different dataset
    max_iterations = get_max_iterations(dataset, classifier, max_time_limit)

    if max_iterations <= 7:
        K = 1
    elif max_iterations > 7 and max_iterations <= 301:
        K = max(1, int((max_iterations - 7) / 6) + 1)
    else:
        K = int((max_iterations - 7 - 49) / 5) + 1

    K_num_list = [K]
    max_acc = []
    exe_time = []
    total_pipe_num = []
    for num in K_num_list:
        start = time.time()
        acc, pipe_number, iter_result, iter_his = \
            run_exp(dataset, classifier, max_time_limit, max_len, seed, num)
        end = time.time()
        max_acc.append(acc)
        total_pipe_num.append(pipe_number)
        exe_time.append(end - start)
        print("Execution time: " + str(end - start))

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

    run_pnas_mlp_no_ensemble(dataset, classifier, max_time_limit, max_len, seed)


