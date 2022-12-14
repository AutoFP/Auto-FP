import time
import numpy as np
import argparse
import warnings

from torch import seed
import smac

from sklearn.metrics import accuracy_score
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

from utils import DATA_DIR, BASE_OUTPUT_DIR, MAX_PIPE_LEN, operator_names
from utils import set_env, load_data, get_pipe, get_model, make_output_dir 

warnings.filterwarnings("ignore")

X_train, X_valid, y_train, y_valid = []
global_start = 0
dataset = ""
classifier = ""
max_time_limit = 0
output_dir = ""
seed = 0

def generate_config_space():
    # Define config space
    cs = smac.configspace.ConfigurationSpace()

    length = CategoricalHyperparameter("length", ["1", "2", "3", "4", "5", "6", "7"], default_value="5")
    cs.add_hyperparameter(length)

    op_len1_1  = CategoricalHyperparameter("op_len1_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    op_len2_1  = CategoricalHyperparameter("op_len2_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len2_2  = CategoricalHyperparameter("op_len2_2", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    op_len3_1  = CategoricalHyperparameter("op_len3_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len3_2  = CategoricalHyperparameter("op_len3_2", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len3_3  = CategoricalHyperparameter("op_len3_3", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    op_len4_1  = CategoricalHyperparameter("op_len4_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len4_2  = CategoricalHyperparameter("op_len4_2", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len4_3  = CategoricalHyperparameter("op_len4_3", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len4_4  = CategoricalHyperparameter("op_len4_4", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    op_len5_1  = CategoricalHyperparameter("op_len5_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len5_2  = CategoricalHyperparameter("op_len5_2", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len5_3  = CategoricalHyperparameter("op_len5_3", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len5_4  = CategoricalHyperparameter("op_len5_4", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len5_5  = CategoricalHyperparameter("op_len5_5", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    op_len6_1  = CategoricalHyperparameter("op_len6_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len6_2  = CategoricalHyperparameter("op_len6_2", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len6_3  = CategoricalHyperparameter("op_len6_3", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len6_4  = CategoricalHyperparameter("op_len6_4", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len6_5  = CategoricalHyperparameter("op_len6_5", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len6_6  = CategoricalHyperparameter("op_len6_6", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    op_len7_1  = CategoricalHyperparameter("op_len7_1", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len7_2  = CategoricalHyperparameter("op_len7_2", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len7_3  = CategoricalHyperparameter("op_len7_3", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len7_4  = CategoricalHyperparameter("op_len7_4", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len7_5  = CategoricalHyperparameter("op_len7_5", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len7_6  = CategoricalHyperparameter("op_len7_6", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")
    op_len7_7  = CategoricalHyperparameter("op_len7_7", ["binarizer", "standardizer", "normalizer",
                                                         "maxabs", "minmax", "power_trans", "quantile_trans"],
                                           default_value="normalizer")

    cs.add_hyperparameters([op_len1_1,
                            op_len2_1, op_len2_2,
                            op_len3_1, op_len3_2, op_len3_3,
                            op_len4_1, op_len4_2, op_len4_3, op_len4_4,
                            op_len5_1, op_len5_2, op_len5_3, op_len5_4, op_len5_5,
                            op_len6_1, op_len6_2, op_len6_3, op_len6_4, op_len6_5, op_len6_6,
                            op_len7_1, op_len7_2, op_len7_3, op_len7_4, op_len7_5, op_len7_6, op_len7_7])

    use_len1_1 = EqualsCondition(child=op_len1_1, parent=length, value="1")

    use_len2_1 = EqualsCondition(child=op_len2_1, parent=length, value="2")
    use_len2_2 = EqualsCondition(child=op_len2_2, parent=length, value="2")

    use_len3_1 = EqualsCondition(child=op_len3_1, parent=length, value="3")
    use_len3_2 = EqualsCondition(child=op_len3_2, parent=length, value="3")
    use_len3_3 = EqualsCondition(child=op_len3_3, parent=length, value="3")

    use_len4_1 = EqualsCondition(child=op_len4_1, parent=length, value="4")
    use_len4_2 = EqualsCondition(child=op_len4_2, parent=length, value="4")
    use_len4_3 = EqualsCondition(child=op_len4_3, parent=length, value="4")
    use_len4_4 = EqualsCondition(child=op_len4_4, parent=length, value="4")

    use_len5_1 = EqualsCondition(child=op_len5_1, parent=length, value="5")
    use_len5_2 = EqualsCondition(child=op_len5_2, parent=length, value="5")
    use_len5_3 = EqualsCondition(child=op_len5_3, parent=length, value="5")
    use_len5_4 = EqualsCondition(child=op_len5_4, parent=length, value="5")
    use_len5_5 = EqualsCondition(child=op_len5_5, parent=length, value="5")

    use_len6_1 = EqualsCondition(child=op_len6_1, parent=length, value="6")
    use_len6_2 = EqualsCondition(child=op_len6_2, parent=length, value="6")
    use_len6_3 = EqualsCondition(child=op_len6_3, parent=length, value="6")
    use_len6_4 = EqualsCondition(child=op_len6_4, parent=length, value="6")
    use_len6_5 = EqualsCondition(child=op_len6_5, parent=length, value="6")
    use_len6_6 = EqualsCondition(child=op_len6_6, parent=length, value="6")

    use_len7_1 = EqualsCondition(child=op_len7_1, parent=length, value="7")
    use_len7_2 = EqualsCondition(child=op_len7_2, parent=length, value="7")
    use_len7_3 = EqualsCondition(child=op_len7_3, parent=length, value="7")
    use_len7_4 = EqualsCondition(child=op_len7_4, parent=length, value="7")
    use_len7_5 = EqualsCondition(child=op_len7_5, parent=length, value="7")
    use_len7_6 = EqualsCondition(child=op_len7_6, parent=length, value="7")
    use_len7_7 = EqualsCondition(child=op_len7_7, parent=length, value="7")

    cs.add_conditions([use_len1_1,
                       use_len2_1, use_len2_2,
                       use_len3_1, use_len3_2, use_len3_3,
                       use_len4_1, use_len4_2, use_len4_3, use_len4_4,
                       use_len5_1, use_len5_2, use_len5_3, use_len5_4, use_len5_5,
                       use_len6_1, use_len6_2, use_len6_3, use_len6_4, use_len6_5, use_len6_6,
                       use_len7_1, use_len7_2, use_len7_3, use_len7_4, use_len7_5, use_len7_6, use_len7_7])
    return cs

def get_prep_pipe(PARAMS):
    pipe_length = int(PARAMS.get('length'))
    temp_pipe = []
    if (pipe_length == 1):
        op_name = PARAMS.get("op_len1_1")
        temp_pipe.append(str(op_name))
    else:
        for i in range(pipe_length):
            op_name = PARAMS.get("op_len" + str(pipe_length) + "_" + str(i+1))
            temp_pipe.append(str(op_name))
    pipe, pipe_str = get_pipe(temp_pipe)
    return pipe, pipe_str

def run(cfg, seed):
    generate_pipe_start = time.time()
    prep_pipe, prep_pipe_str = get_prep_pipe(cfg.get_dictionary())
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
    f = open(f'{output_dir}/smac_wallock_{seed}.csv', 'a')
    f.write(str(global_mid - global_start) + "\n")
    f = open(f'{output_dir}/smac_pipe_{seed}.csv', 'a')
    f.write(prep_pipe_str + "\n")
    f = open(f'{output_dir}/smac_score_{seed}.csv', 'a')
    f.write(str(score) + "\n")
    f = open(f'{output_dir}/smac_eval_time_{seed}.csv', 'a')
    f.write(str(generate_pipe_end - generate_pipe_start) + "," +
            str(prep_train_end - prep_train_start) + "," +
            str(prep_valid_end - prep_valid_start) + "," +
            str(train_end - train_start) + "," +
            str(predict_end - predict_start) + "," +
            str(eval_score_end - eval_score_start) + "\n")

    return -score

def run_smac(ds, clf, maxtl, max_len, local_seed):
    global X_train, X_valid, y_train, y_valid
    global global_start
    global dataset
    global classifier
    global max_time_limit
    global output_dir
    global seed

    seed = local_seed

    dataset = ds
    classifier = clf
    max_time_limit = maxtl
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='SMAC')

    X_train, X_valid, y_train, y_valid = load_data(dataset)
    cs = generate_config_space()
    global_start = time.time()
    
    scenario = Scenario({"run_obj": "quality",  
                         "wallclock_limit": max_time_limit,  
                         "cs": cs, 
                         "deterministic": False,
                         "output_dir_for_this_run": None,
                         "output_dir": None,
                        })

    #[0,42,137,498,1479]
    smac = SMAC4HPO(
        scenario=scenario, 
        rng=np.random.RandomState(seed),
        tae_runner=run
    )

    try:
        incumbent = smac.optimize(dataset=dataset, classifier=classifier, max_time_limit=max_time_limit)
    finally:
        incumbent = smac.solver.incumbent


