from GFLC import GFLC
from aequitas.flow.experiment import Experiment


import pickle
import io
import torch
import datetime
import hashlib
import json
import pickle
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from sklearn.metrics import classification_report



def feature_suppression2(X: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    """
    Suppresses the sensitive feature from the DataFrame if it exists.
    Assumes unawareness_features is True, so the column corresponding
    to the sensitive attribute (s.name) is dropped.

    Parameters:
        X (pd.DataFrame): The input DataFrame.
        s (pd.Series): The sensitive attribute as a Series. Its name is used to identify the column.

    Returns:
        pd.DataFrame: The DataFrame with the sensitive feature removed (if present).
    """
    X_transformed = X.copy()
    if s.name in X_transformed.columns:
        X_transformed = X_transformed.drop(columns=[s.name])
    return X_transformed



dataset_name = "BankAccountFraud"
variant = "TypeII"
method_name = "Fair-OBNC"
experiment_name = "noise_injection_experiment"
config_file_path = Path("path to your yaml file")


class SimpleExperiment(Experiment):
    def run(self):
        """Override run to return only dataset splits."""
        self._read_datasets()
        
        # Get the first dataset
        for dataset_name, dataset in self.datasets.items():
            return {
                "X_train": dataset["train"]["X"].copy(deep=True),
                "y_train": dataset["train"]["y"].copy(deep=True),
                "s_train": dataset["train"]["s"].copy(deep=True),
                "X_val": dataset["validation"]["X"].copy(deep=True),
                "y_val": dataset["validation"]["y"].copy(deep=True),
                "s_val": dataset["validation"]["s"].copy(deep=True),
                "X_test": dataset["test"]["X"].copy(deep=True),
                "y_test": dataset["test"]["y"].copy(deep=True),
                "s_test": dataset["test"]["s"].copy(deep=True),
            }
        
        return {}
    

experiment = SimpleExperiment(
    config_file=config_file_path,
    name=experiment_name
)

splits = experiment.run()


X_train=splits['X_train']
y_train=splits['y_train']
s_train=splits['s_train']
X_val=splits['X_train']
y_val=splits['y_val']
s_val=splits['s_val']
X_test=splits['X_test']
y_test=splits['y_test']
s_test=splits['s_test']


X_transformed = feature_suppression2(X_train, s_train)
X_transformed = pd.get_dummies(X_transformed)
X_transformed_numpy_array = np.array(X_transformed.values, dtype=np.float32)
y_train0 = np.array(y_train.values, dtype=np.float64)
s_train_numeric = s_train.cat.codes
s_train_numeric_numpy = np.array(s_train_numeric.values, dtype=np.int64)


X_train1 = X_transformed
y_train1 = y_train0
s_train1 = s_train_numeric_numpy




# Usage
gflc = GFLC(
    k=10,
    ricci_iter=2,
    alpha=0.3,
    beta=0.5,
    gamma=0.2,
    pos_threshold=0.25,
    neg_threshold=0.85,
    max_fpr=0.03
)


# FIT
start_time = time.time()
gflc.fit(X_train1, y_train1, s_train1)  # Replace with your data variables
end_time = time.time()
print(f"Completed in {end_time - start_time:.2f} seconds")



# Correct Labels
start_time = time.time()
y_corrected = gflc.correct_labels(
    X_train1, y_train1, s_train1,
    disparity_target=0.05
)
end_time = time.time()
print(type(y_corrected))
print(y_corrected.shape)
print(classification_report(y_train1, y_corrected, target_names=['0', '1']))





