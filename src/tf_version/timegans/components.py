# Basic imports
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn related import
import sklearn
from sklearn.preprocessing import MinMaxScaler

# Tensorflow related imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Parameters
hidden_dim = 24
num_layers = 3

# Logger
results_path = Path("results")
log_dir = results_path / f"experiment_{0:02}"
writer = tf.summary.create_file_writer(log_dir.as_posix())


def get_input_placeholders(dataset="batadal"):
    seq_len = 24
    X = None
    Z = None
    if dataset == "batadal":
        n_seq = 26
        X = Input(shape=[seq_len, n_seq], name="RealData")
        Z = Input(shape=[seq_len, n_seq], name="RandomData")
    elif dataset == "swat":
        n_seq = 18
        X = Input(shape=[seq_len, n_seq], name="RealData")
        Z = Input(shape=[seq_len, n_seq], name="RandomData")
    else:
        print("Please check your 'dataset' hyperparameter again.")
        return
    return X, Z


def make_rnn(n_layers, hidden_units, output_units, name):
    """Build RNN Block Generator

    Args:
        n_layers (_type_): _description_
        hidden_units (_type_): _description_
        output_units (_type_): _description_
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    return Sequential(
        [
            GRU(units=hidden_units, return_sequences=True, name=f"GRU_{i + 1}")
            for i in range(n_layers)
        ]
        + [Dense(units=output_units, activation="sigmoid", name="OUT")],
        name=name,
    )
