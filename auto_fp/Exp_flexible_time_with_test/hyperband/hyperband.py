import os
import time
import numpy as np
import argparse
import warnings
import ConfigSpace as CS
import hpbandster.core.nameserver as hpns

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from hpbandster.optimizers import HyperBand
from hpbandster.core.worker import Worker
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

from utils import MAX_PIPE_LEN
from utils import set_env, load_data, get_pipe, make_output_dir 

warnings.filterwarnings("ignore")

class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.seed = kwargs['seed']
        self.dataset = kwargs['dataset']
        self.classifier = kwargs['classifier']
        self.max_time_limit = kwargs['max_time_limit']
        self.max_budget = kwargs['max_budget']
        self.X_train = kwargs['X_train']
        self.X_valid = kwargs['X_valid']
        self.y_train = kwargs['y_train']
        self.y_valid = kwargs['y_valid']
        self.global_start = kwargs['global_start']
        self.output_dir = kwargs['output_dir']

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        generate_pipe_start = time.time()
        prep_pipe, prep_pipe_str = self.get_pipe(config)
        generate_pipe_end = time.time()

        model = self.get_model(int(budget))
      
        prep_train_start, prep_train_end = 0, 0
        prep_valid_start, prep_valid_end = 0, 0
        train_start, train_end = 0, 0
        predict_start, predict_end = 0, 0
        eval_score_start, eval_score_end = 0, 0
        time_limit_reached = False

        try:
            prep_train_start = time.time()
            X_train_new = prep_pipe.fit_transform(self.X_train)
            prep_train_end = time.time()

            prep_valid_start = time.time()
            X_valid_new = prep_pipe.transform(self.X_valid)
            prep_valid_end = time.time()

            train_start = time.time()
            model.fit(np.array(X_train_new), np.array(self.y_train))
            train_end = time.time()

            predict_start = time.time()
            y_pred = model.predict(np.array(X_valid_new))
            predict_end = time.time()

            eval_score_start = time.time()
            score = accuracy_score(self.y_valid, y_pred)
            eval_score_end = time.time()

        except:
            score = 0
        global_mid = time.time()
        if (global_mid - self.global_start) >= self.max_time_limit:
            time_limit_reached = True
        if (budget == self.max_budget):
            f = open(f'{self.output_dir}/hyperband_wallock_{self.seed}.csv', 'a')
            f.write(str(global_mid - self.global_start) + "\n")
            f = open(f'{self.output_dir}/hyperband_pipe_{self.seed}.csv', 'a')
            f.write(prep_pipe_str + "\n")
            f = open(f'{self.output_dir}/hyperband_score_{self.seed}.csv', 'a')
            f.write(str(score) + "\n")
            f = open(f'{self.output_dir}/hyperband_eval_time_{self.seed}.csv', 'a')
            f.write(str(generate_pipe_end - generate_pipe_start) + "," +
                    str(prep_train_end - prep_train_start) + "," +
                    str(prep_valid_end - prep_valid_start) + "," +
                    str(train_end - train_start) + "," +
                    str(predict_end - predict_start) + "," +
                    str(eval_score_end - eval_score_start) + "\n")
        
        if time_limit_reached:
            os._exit(0)
        
        return({
                    'loss': 1 - float(score),  # this is the a mandatory field to run hyperband
                    'info': {'accuracy': float(score)}  # can be used for any user-defined information - also mandatory
                })

    def get_pipe(self, PARAMS):
        pipe_length = int(PARAMS.get('length'))
        temp_pipe = []
        if (pipe_length == 1):
            op_name = PARAMS.get("op_len1_1")
            temp_pipe.append(str(op_name))
        else:
            for i in range(pipe_length):
                op_name = PARAMS.get("op_len" + str(pipe_length) + "_" + str(i+1))
                temp_pipe.append(op_name)
        pipe, pipe_str = get_pipe(temp_pipe)
        return pipe, pipe_str

    def get_model(self, budget):
        if self.classifier == 'LR':
            model = LogisticRegression(random_state=0, n_jobs=1, max_iter=budget)
        elif self.classifier == 'XGB':
            model =  XGBClassifier(random_state=0, nthread=1, n_jobs=1, n_estimators=budget)
        elif self.classifier == 'MLP':
            model = MLPClassifier(random_state=0, max_iter=budget)
        return model

    @staticmethod
    def get_configspace(seed):
        cs = CS.ConfigurationSpace()
        cs.seed(seed)
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
        return(cs)

def get_budget_and_port_num(classifier):
    if classifier == "LR":
        max_budget = 100
        port_num = 8913
    elif classifier == "XGB":
        max_budget = 100
        port_num = 8914
    elif classifier == "MLP":
        max_budget = 200
        port_num = 8915
    return max_budget, port_num

def run_hyperband(dataset, classifier, max_time_limit, max_len, seed):
    output_dir = make_output_dir(dataset, classifier, max_time_limit, algorithm='HYPERBAND')

    np.random.seed(seed)

    min_budget = 1
    max_budget = 1
    port_num = 0
    max_budget, port_num = get_budget_and_port_num(classifier)
    n_iterations = 1000000
    
    X_train, X_valid, y_train, y_valid = load_data(dataset)
    global_start = time.time()
    params = {
        'seed': seed,
        'dataset': dataset,
        'classifier': classifier,
        'max_time_limit': max_time_limit,
        'max_budget': max_budget,
        'X_train': X_train,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_valid': y_valid,
        'global_start': global_start,
        'output_dir': output_dir
    }

    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=port_num)
    NS.start()  
    w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1', nameserver_port=port_num, run_id='example1', **params)
    w.run(background=True)
    hyperband = HyperBand(  
        configspace = w.get_configspace(seed),
        run_id = 'example1', 
        nameserver='127.0.0.1',
        nameserver_port=port_num,
        min_budget=min_budget, 
        max_budget=max_budget,
        eta=3
    )
    hyperband.run(n_iterations=n_iterations)

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

    run_hyperband(dataset, classifier, max_time_limit, max_len, seed)
