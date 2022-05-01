# Basic imports
from concurrent.futures import process
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import datasets
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

# Local imports
from utils.preprocessing import (
    preprocess_df,
    plot_df,
    corr_plot,
    normalize_df,
    rolling_window,
)
from utils.preprocessing import iid_converter
from timegans.components import make_rnn

# Set device
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using CUDA")
else:
    print("Using CPU")

# Set seaborn style
sns.set_style("white")

# Set results path
results_path = Path("results")
log_dir = results_path / f"experiment_{0:02}"
if not log_dir.exists():
    log_dir.mkdir(parents=True)

dataset = "batadal"  # swat


def main():
    processed_df, tickers = preprocess_df(dataset)
    # plot_df(processed_df, tickers)
    corr_plot(processed_df)
    scaler, scaled_data = normalize_df(processed_df)
    data, n_windows = rolling_window(processed_df, scaled_data)

    # get iid data
    real_series_iter, random_series = iid_converter(data, n_windows, dataset)

    hidden_dim = 24
    num_layers = 3
    n_seq = 26

    # generate building block of Time GAN
    embedder = make_rnn(
        n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Embedder"
    )
    recovery = make_rnn(
        n_layers=3, hidden_units=hidden_dim, output_units=n_seq, name="Recovery"
    )


if __name__ == "__main__":
    main()
