import time
import sys
import random
import numpy as np
import argparse
import warnings

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, \
                                  MaxAbsScaler, MinMaxScaler, Binarizer, \
                                  PowerTransformer, QuantileTransformer

from utils import DATA_DIR, BASE_OUTPUT_DIR, MAX_PIPE_LEN, preprocessors_dic, operator_names
from utils import set_env, load_data, get_pipe, get_model, make_output_dir_extended_space

warnings.filterwarnings("ignore")

# Extra parameters in extended space
#binarizer_threshold = np.arange(0, 1.01, 0.05).tolist()
binarizer_threshold = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
normalizer_norm = ['l1', 'l2', 'max']
standard_with_mean = [True, False]
power_standardize = [True, False]
#quantile_n_quantiles = np.arange(10, 2001, 1).tolist()
quantile_n_quantiles = [10, 100, 200, 500, 1000, 1200, 1500, 2000]
quantile_distribution = ['uniform', 'normal']

def mutate(parent, mutate_way_seed, op_seed, pos_seed):
    mutations = []
    if (len(parent) == 1):
        mutations = ["add", "replace"]
    elif (len(parent) == max_len):
        mutations = ["delete", "replace", "switch"]
    else:
        mutations = ["add", "delete", "replace", "switch"]
    #random.seed(1440)
    np.random.seed(mutate_way_seed)
    mutate_type = np.random.choice(mutations)
    #print(mutate_type)

    child = []
    #print(parent)
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

def perturbation(top_pipe, resample_probablity,
                 resample_seed, mutate_way_seed, op_seed, pos_seed):
    result_pipe = []
    np.random.seed(resample_seed)
    prob = np.random.random()
    if prob < resample_probablity:
        ''' Do resampling '''
        np.random.seed(resample_seed)
        length = np.random.randint(1, 7)
        for i in range(length):
            result_pipe.append(operator_names[np.random.randint(1, len(operator_names)) - 1])
    else:
        ''' Do perturbation (small mutation) '''
        result_pipe = mutate(top_pipe, mutate_way_seed, op_seed, pos_seed)
    return result_pipe

def exploit_and_explore(population, bot_trial_info, top_trial_info, resample_probability,
                        resample_seed, mutate_way_seed, op_seed, pos_seed):
    result_pipe = []
    bot_index = population.index(bot_trial_info)
    top_index = population.index(top_trial_info)
    bot_pipe = bot_trial_info[0]
    top_pipe = top_trial_info[0]

    result_pipe = perturbation(top_pipe, resample_probability,
                               resample_seed, mutate_way_seed, op_seed, pos_seed)
    return result_pipe

# Generate the new params of next searching pipeline (need to consider random seed)
def generate_new_param(b_thresh_seed, n_norm_seed,
                      s_with_mean_seed, p_standardize_seed,
                      q_n_quantiles_seed, q_distribution_seed):
    np.random.seed(b_thresh_seed)
    b_thresh = np.random.choice(binarizer_threshold, size=1, replace=True)[0]
    
    np.random.seed(n_norm_seed)
    n_norm = np.random.choice(normalizer_norm, size=1, replace=True)[0]
    
    np.random.seed(s_with_mean_seed)
    s_with_mean = np.random.choice(standard_with_mean, size=1, replace=True)[0]
    
    np.random.seed(p_standardize_seed)
    p_standardize = np.random.choice(power_standardize, size=1, replace=True)[0]
    
    np.random.seed(q_n_quantiles_seed)
    q_n_quantiles = np.random.choice(quantile_n_quantiles, size=1, replace=True)[0]
    
    np.random.seed(q_distribution_seed)
    q_distribution = np.random.choice(quantile_distribution, size=1, replace=True)[0]
    
    global preprocessors_dic
    global operator_names
    preprocessors_dic = {}
    operator_names = []
    
    operator_names.append(f"binarizer_threshold_{str(b_thresh)}")
    preprocessors_dic[f"binarizer_threshold_{str(b_thresh)}"] = Binarizer(threshold=b_thresh)
    
    operator_names.append(f"normalizer_norm_{n_norm}")
    preprocessors_dic[f"normalizer_norm_{n_norm}"] = Normalizer(norm=n_norm)
    
    operator_names.append(f"standard_with_mean_{str(s_with_mean)}")
    preprocessors_dic[f"standard_with_mean_{str(s_with_mean)}"] = StandardScaler(with_mean=s_with_mean)
    
    operator_names.append(f"power_standardize_{str(p_standardize)}")
    preprocessors_dic[f"power_standardize_{str(p_standardize)}"] = PowerTransformer(standardize=p_standardize)
    
    operator_names.append(f"quantile_n_quantiles_{str(q_n_quantiles)}")
    preprocessors_dic[f"quantile_n_quantiles_{str(q_n_quantiles)}"] = QuantileTransformer(n_quantiles=q_n_quantiles, random_state=0)
    
    operator_names.append(f"quantile_distribution_{str(q_distribution)}")
    preprocessors_dic[f"quantile_distribution_{str(q_distribution)}"] = QuantileTransformer(output_distribution=q_distribution, random_state=0)
    
def run_pbt_extended_space_with_heuristic(dataset, classifier, max_time_limit, max_len, seed):
    output_dir = make_output_dir_extended_space(\
        dataset, classifier, max_time_limit, \
        algorithm='PBT', type=('balanced', 'two-step'))

    # [0, 42, 167,578,1440]
    all_list = [i for i in range(0, 5000000)]
    np.random.seed(seed)
    top_choice_seeds = np.random.choice(all_list, 2000000, replace=False)
    resample_seeds = np.random.choice(all_list, 2000000, replace=False)
    mutate_way_seeds = np.random.choice(all_list, 2000000, replace=False)
    op_seeds = np.random.choice(all_list, 2000000, replace=False)
    pos_seeds = np.random.choice(all_list, 2000000, replace=False)

    #seeds for generating parameters
    b_thresh_seeds = np.random.choice(all_list, 500, replace=False)
    n_norm_seeds = np.random.choice(all_list, 500, replace=False)
    s_with_mean_seeds = np.random.choice(all_list, 500, replace=False)
    p_standardize_seeds = np.random.choice(all_list, 500, replace=False)
    q_n_quantiles_seeds = np.random.choice(all_list, 500, replace=False)
    q_distribution_seeds = np.random.choice(all_list, 500, replace=False)

    # Begin PBT code
    population_size = 100
    fraction = 0.2
    resample_probability = 0.25
    epochs = 1000000
    time_for_each_param = 60

    X_train, X_valid, y_train, y_valid = load_data()
    global_start = time.time()

    global_time_limit_reached = False

    count = -1 # control the used seeds, promise in each step the seeds are different and experiment replication
    num_explored_param = -1

    while True:
        # Initialize population
        population = []
        # Randomly generate a group of parameters and change the value of preprocessors_dic
        num_explored_param += 1
        generate_new_param(b_thresh_seeds[num_explored_param], n_norm_seeds[num_explored_param],
                           s_with_mean_seeds[num_explored_param], p_standardize_seeds[num_explored_param],
                           q_n_quantiles_seeds[num_explored_param], q_distribution_seeds[num_explored_param])
        print(preprocessors_dic)
        print(operator_names)
        param_start = time.time()
        param_time_limit_reached = False

        #Generate random pipelines, which equal to population size
        while (len(population) < population_size):
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
        
            # Judge whether the time limit of this parameter reached
            param_mid = time.time()
            if (param_mid - param_start) >= time_for_each_param:
                param_time_limit_reached = True
        
            # Judge whether the time limit of global process reached
            global_mid = time.time()
            if (global_mid - global_start) >= max_time_limit:
                global_time_limit_reached = True
            
            f = open(f'{output_dir}/pbt_pick_time_1.csv', 'a')
            f.write(str(pick_time) + "\n")
            f = open(f'{output_dir}/pbt_wallock_1.csv', 'a')
            f.write(str(global_mid - global_start) + "\n")
            f = open(f'{output_dir}/pbt_pipe_1.csv', 'a')
            f.write(prep_pipe_str + "\n")
            f = open(f'{output_dir}/pbt_score_1.csv', 'a')
            f.write(str(score) + "\n")
            f = open(f'{output_dir}/pbt_eval_time_1.csv', 'a')
            f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                    str(prep_train_end - prep_train_start) + "," +
                    str(prep_valid_end - prep_valid_start) + "," +
                    str(train_end - train_start) + "," +
                    str(pred_end - pred_start) + "," +
                    str(eval_score_end - eval_score_start) + "\n")
        
            if global_time_limit_reached:
                sys.exit()
            elif param_time_limit_reached:
                break

            population.append((temp_pipe, score))
    
        # Contining find next parameter if time for current parameter is reached
        if param_time_limit_reached:
            continue
        
        # Using PBT framework to do pipeline searching
        current_epoch = 0
        while current_epoch < epochs:
            current_epoch += 1
            temp_population = []
            #sort population based on accuracy score
            population.sort(key=lambda x : x[1], reverse=True)
            cutoff = int(np.ceil(fraction * len(population)))
            tops = population[:cutoff]
            bottoms = population[len(population) - cutoff:]
            for bottom in bottoms:
                score = 0
                exploit_and_explore_pipe = []
                while score == 0:
                    count += 1
                    pick_start = time.time()
                    np.random.seed(top_choice_seeds[count])
                    top_idx = [idx for idx in range(len(tops))]
                    top = tops[np.random.choice(top_idx)]
                    exploit_and_explore_pipe = exploit_and_explore(population, bottom, top, resample_probability,
                                                                   resample_seed=resample_seeds[count],
                                                                   mutate_way_seed=mutate_way_seeds[count],
                                                                   op_seed=op_seeds[count],
                                                                   pos_seed=pos_seeds[count])
                    pick_end = time.time()
                    pick_time = pick_end - pick_start

                    # begin evaluation
                    temp_pipe = []
                    for i in range(length):
                        temp_pipe.append(exploit_and_explore_pipe[i])
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
                
                    # Judge whether the time limit of this parameter reached
                    param_mid = time.time()
                    if (param_mid - param_start) >= time_for_each_param:
                        param_time_limit_reached = True
        
                    # Judge whether the time limit of global process reached
                    global_mid = time.time()
                    if (global_mid - global_start) >= max_time_limit:
                        global_time_limit_reached = True
                    
                    f = open(f'{output_dir}/pbt_pick_time_1.csv', 'a')
                    f.write(str(pick_time) + "\n")
                    f = open(f'{output_dir}/pbt_wallock_1.csv', 'a')
                    f.write(str(global_mid - global_start) + "\n")
                    f = open(f'{output_dir}/pbt_pipe_1.csv', 'a')
                    f.write(prep_pipe_str + "\n")
                    f = open(f'{output_dir}/pbt_score_1.csv', 'a')
                    f.write(str(score) + "\n")
                    f = open(f'{output_dir}/pbt_eval_time_1.csv', 'a')
                    f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                            str(prep_train_end - prep_train_start) + "," +
                            str(prep_valid_end - prep_valid_start) + "," +
                            str(train_end - train_start) + "," +
                            str(pred_end - pred_start) + "," +
                            str(eval_score_end - eval_score_start) + "\n")
                
                    if global_time_limit_reached:
                        sys.exit()
                    elif param_time_limit_reached:
                        break
                    
                population[population.index(bottom)] = (exploit_and_explore_pipe, score)
                if param_time_limit_reached:
                    break
            if param_time_limit_reached:
                break

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

    run_pbt_extended_space_with_heuristic(dataset, classifier, max_time_limit, max_len, seed)


        