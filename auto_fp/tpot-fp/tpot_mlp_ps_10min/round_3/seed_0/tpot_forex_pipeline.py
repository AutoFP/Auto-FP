import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, Normalizer, StandardScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=0)

# Average CV score on the training set was: 0.6230209670517758
exported_pipeline = make_pipeline(
    StandardScaler(with_mean=True),
    Normalizer(norm="l2"),
    MaxAbsScaler(),
    MLPClassifier()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
