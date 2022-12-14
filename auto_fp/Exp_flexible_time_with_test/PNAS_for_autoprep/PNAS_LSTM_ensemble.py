import time
import sys
import random
import numpy as np
import argparse
import warnings
import torch
import torch.nn.init as init

from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from utils import DATA_DIR, BASE_OUTPUT_DIR, MAX_PIPE_LEN, operator_names
from utils import set_env, load_data, get_pipe, get_model, make_output_dir 

warnings.filterwarnings("ignore")

X_train, X_valid, y_train, y_valid = load_data()

def get_max_iterations(dataset, classifier, max_time_limit):
    result = 0
    if classifier == 'LR':
        if dataset == "austrilian":
            result = 700
        if dataset == "blood":
            result = 600
        if dataset == "emotion":
            result = 700
        if dataset == "forex":
            result = 600
        if dataset == "heart":
            result = 700
        if dataset == "jasmine":
            result = 900
        if dataset == "madeline":
            result = 500
        if dataset == "pd_speech_features":
            result = 450
        if dataset == "wine_quality":
            result = 650
        if dataset == "thyroid-allhyper":
            result = 650
    if classifier == 'XGB':
        if dataset == "austrilian":
            result = 700
        if dataset == "blood":
            result = 700
        if dataset == "emotion":
            result = 650
        if dataset == "forex":
            result = 350
        if dataset == "heart":
            result = 700
        if dataset == "jasmine":
            result = 450
        if dataset == "madeline":
            result = 250
        if dataset == "pd_speech_features":
            result = 350
        if dataset == "wine_quality":
            result = 400
        if dataset == "thyroid-allhyper":
            result = 500
    if classifier == 'MLP':
        if dataset == "austrilian":
            result = 650
        if dataset == "blood":
            result = 650
        if dataset == "emotion":
            result = 600
        if dataset == "forex":
            result = 150
        if dataset == "heart":
            result = 650
        if dataset == "jasmine":
            result = 300
        if dataset == "madeline":
            result = 230
        if dataset == "pd_speech_features":
            result = 300
        if dataset == "wine_quality":
            result = 300
        if dataset == "thyroid-allhyper":
            result = 400
    result = int(result * (max_time_limit / float(3600)))
    return result

def embedding(structure):
    return (np.sum(structure, axis=0) / (len(structure))).tolist()

def one_hot(structure):
    encode = np.zeros((len(structure), len(operator_names)))
    for i in range(len(structure)):
        encode[i][structure[i]] = 1
    return encode.tolist()

class LSTM_predictor(torch.nn.Module):
    def __init__(self, embedding_dim = 7, hidden_dim = 30, num_layers = 2, out_dim=1, seed=0):
        super(LSTM_predictor,self).__init__()
        torch.random.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(len(operator_names), embedding_dim)
        self.embedding.weight.data.uniform_(-1e5, 1e5)
        #init.xavier_normal_(self.embedding.weight.data, gain=2)
        print(self.embedding.weight.data)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.dense = torch.nn.Linear(hidden_dim, out_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.hidden = self.init_hidden()

        self.seed = seed

    def init_hidden(self):
        torch.random.manual_seed(self.seed)
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))

    def forward(self,x):
        embeds = self.embedding(x)
        #output, (h,c) = self.lstm(embeds.view(len(x), 1, -1))
        output, self.hidden = self.lstm(embeds.view(len(x), 1, -1), self.hidden)
        output1 = self.dense(output)
        output2 = self.sigmoid(output1)
        return output2

def run_exp(dataset, classifier, max_time_limit, max_len, seed, K_num = 50):
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='PNAS_LSTM_ENSEMBLE')

    # [0, 42, 167,578,1440]
    random.seed(seed)
    np.random.seed(seed)

    X_train, X_valid, y_train, y_valid = load_data(dataset)
    global_start = time.time()
    print("K_NUM = " + str(K_num))

    # History pipes
    history = []
    no_encode_history = []
    validate_acc =[]
    pipelines = []

    # Predictor
    predictors = [LSTM_predictor(seed=seed), LSTM_predictor(seed=seed), \
        LSTM_predictor(seed=seed), LSTM_predictor(seed=seed), LSTM_predictor(seed=seed)]

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
            no_encode_history.append([i])

        except:
            score = 0
        global_mid = time.time()
        if global_mid - global_start >= max_time_limit:
            time_limit_reached = True
        else:
            time_limit_reached = False
        f = open(f'{output_dir}/pnas_lstm_ensemble_pick_time_{seed}.csv', 'a')
        f.write(str(pick_time) + "\n")
        f = open(f'{output_dir}/pnas_lstm_ensemble_wallock_{seed}.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'{output_dir}/pnas_lstm_ensemble_pipe_{seed}.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{output_dir}/pnas_lstm_ensemble_score_{seed}.csv', 'a')
        f.write(str(score) + "\n")
        f = open(f'{output_dir}/pnas_lstm_ensemble_eval_time_{seed}.csv', 'a')
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

    # Split data into K-fold
    update_start = time.time()
    kf = KFold(n_splits=5)
    predict_step = -1
    for train_index, test_index in kf.split(no_encode_history):
        predict_step += 1
        print("TRAIN:", train_index, "TEST:", test_index)
        temp_no_encode_history = []
        temp_validate_acc = []
        for item in train_index:
            temp_no_encode_history.append(no_encode_history[item])
            temp_validate_acc.append(validate_acc[item])
        temp_no_encode_history_tensor = Variable(torch.from_numpy(np.array(temp_no_encode_history)).type(torch.LongTensor))
        temp_validate_acc_tensor = Variable(torch.from_numpy(np.array(temp_validate_acc)).type(torch.FloatTensor))
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(predictors[predict_step].parameters(), lr=0.01)

        for epoch in range(100):
            for train_idx in range(len(temp_no_encode_history_tensor)):
                predictors[predict_step].zero_grad()
                predictors[predict_step].hidden = predictors[predict_step].init_hidden()
                y_pred = predictors[predict_step](temp_no_encode_history_tensor[train_idx])
                loss = criterion(y_pred, temp_validate_acc_tensor[train_idx])
                loss.backward()
                optimizer.step()
            print(epoch)
    update_end = time.time()

    # Start main exp
    K = K_num

    temp_pipes = [history.copy()]
    temp_acc = [validate_acc.copy()]

    for length in range(max_len - 1):
        pick_start = time.time()
        generate_pipes = []
        generate_acc = []
        for i in range(len(temp_pipes[-1])):
            pre_pipe = temp_pipes[-1][i].copy()
            for j in range(len(operator_names)):
                generate_pipes.append(pre_pipe + [j])
                generate_acc.append(sum(predictor(Variable(torch.from_numpy(np.array(pre_pipe + [j])).type(torch.LongTensor)))[0][0][0] for predictor in predictors) / len(predictors))

        # Get top-k result
        topK_pipe = []
        topK_acc = []
        temp_topK_pipe = []

        if (K > len(generate_pipes)):
            temp_topK_pipe = generate_pipes.copy()
        else:
            temp_topK_pipe = []
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
            if global_mid - global_start >= max_time_limit:
                time_limit_reached = True
            else:
                time_limit_reached = False
            f = open(f'{output_dir}/pnas_lstm_ensemble_pick_time_{seed}.csv', 'a')
            f.write(str(pick_time) + "\n")
            f = open(f'{output_dir}/pnas_lstm_ensemble_wallock_{seed}.csv', 'a')
            f.write(str(global_mid - global_start) + "\n")
            f = open(f'{output_dir}/pnas_lstm_ensemble_pipe_{seed}.csv', 'a')
            f.write(prep_pipe_str + "\n")
            f = open(f'{output_dir}/pnas_lstm_ensemble_score_{seed}.csv', 'a')
            f.write(str(score) + "\n")
            f = open(f'{output_dir}/pnas_lstm_ensemble_eval_time_{seed}.csv', 'a')
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
        temp_pipes.append(topK_pipe)
        temp_acc.append(topK_acc)

        # Update predictor
        update_start = time.time()
        history += topK_pipe
        validate_acc += topK_acc
        for i in range(len(topK_pipe)):
            no_encode_history.append(topK_pipe[i])

        kf = KFold(n_splits=5)
        predict_step = -1
        for train_index, test_index in kf.split(no_encode_history):
            predict_step += 1
            print("TRAIN:", train_index, "TEST:", test_index)
            temp_no_encode_history = []
            temp_validate_acc = []
            for item in train_index:
                temp_no_encode_history.append(no_encode_history[item])
                temp_validate_acc.append(validate_acc[item])
            temp_validate_acc_tensor = Variable(torch.from_numpy(np.array(temp_validate_acc)).type(torch.FloatTensor))
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(predictors[predict_step].parameters(), lr=0.01)

            temp_predictor = predictors[predict_step]
            for epoch in range(100):
                for train_idx in range(len(temp_no_encode_history)):
                    temp_predictor.zero_grad()
                    temp_predictor.hidden = temp_predictor.init_hidden()
                    y_pred = temp_predictor(Variable(torch.from_numpy(np.array(temp_no_encode_history[train_idx])).type(torch.LongTensor)))
                    optimizer.zero_grad()
                    loss = criterion(y_pred, temp_validate_acc_tensor[train_idx])
                    loss.backward()
                    optimizer.step()
                print(epoch)
        update_end = time.time()
        print(length)

    print("Max acc: " + str(max(validate_acc)))
    print("Total pipe num: " + str(len(validate_acc)))
    return max(validate_acc), len(validate_acc), validate_acc, history

def run_pnas_lstm_ensemble(dataset, classifier, max_time_limit, max_len, seed): 
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
    iter_his = []
    iter_result = []
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

    run_pnas_lstm_ensemble(dataset, classifier, max_time_limit, max_len, seed)