import warnings
import time
import sys
import random
import numpy as np
import argparse
import warnings
import torch

from sklearn.metrics import accuracy_score
from torch.distributions import Categorical

from utils import DATA_DIR, BASE_OUTPUT_DIR, MAX_PIPE_LEN, operator_names
from utils import set_env, load_data, get_pipe, get_model, make_output_dir 
from Policy import Policy, ExponentialMovingAverage

warnings.filterwarnings("ignore")

def select_action(policy):
    probs = policy()
    m = Categorical(probs)
    action = m.sample()
    #policy.saved_log_probs.append(m.log_prob(action))
    return m.log_prob(action), action.cpu().tolist()

def run_reinforce(dataset, classifier, max_time_limit, max_len, seed):
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='REINFORCE')

    # [0, 42, 167,578,1440]
    random.seed(seed)
    torch.manual_seed(seed)
    learning_rate = 0.02
    EMA_momentum= 0.9
    RL_steps = 1050000

    policy    = Policy(max_len, operator_names)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    eps       = np.finfo(np.float32).eps.item()
    baseline  = ExponentialMovingAverage(EMA_momentum)

    update_start = None
    update_end = None

    X_train, X_valid, y_train, y_valid = load_data(dataset)
    global_start = time.time()

    time_limit_reached = False

    for istep in range(RL_steps):
        pick_start = time.time()
        log_prob, action = select_action( policy )
        arch = policy.generate_arch( action )
        pick_end = time.time()
        if update_start is None:
            pick_time = pick_end - pick_start
        else:
            pick_time = (pick_end - pick_start) + (update_end - update_start)
        temp_pipe = []
        for i in range(len(arch)):
            temp_pipe.append(operator_names[arch[i]])

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
        except:
            score = 0

        global_mid = time.time()
        if (global_mid - global_start) >= max_time_limit:
            time_limit_reached = True
        f = open(f'{output_dir}/reinforce_pick_time_{seed}.csv', 'a')
        f.write(str(pick_time) + "\n")
        f = open(f'{output_dir}/reinforce_wallock_{seed}.csv', 'a')
        f.write(str(global_mid - global_start) + "\n")
        f = open(f'{output_dir}/reinforce_pipe_{seed}.csv', 'a')
        f.write(prep_pipe_str + "\n")
        f = open(f'{output_dir}/reinforce_score_{seed}.csv', 'a')
        f.write(str(score) + "\n")
        f = open(f'{output_dir}/reinforce_eval_time_{seed}.csv', 'a')
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

        update_start = time.time()
        reward = score
        baseline.update(reward)
        # calculate loss
        policy_loss = ( -log_prob * (reward - baseline.value()) ).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        update_end = time.time()
        print(istep)

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

    run_reinforce(dataset, classifier, max_time_limit, max_len, seed)
    