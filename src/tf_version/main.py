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
    iid_converter,
)
from timegans.components import get_input_placeholders, make_rnn
from timegans.train import (
    train_autoencoder_init,
    get_generator_moment_loss,
    train_generator,
)

# Logger
results_path = Path("results")
log_dir = results_path / f"experiment_{0:02}"
writer = tf.summary.create_file_writer(log_dir.as_posix())

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


def main():
    # 1. set dataset
    dataset = "batadal"  # swat

    processed_df, tickers = preprocess_df(dataset)
    # plot_df(processed_df, tickers)
    # corr_plot(processed_df)
    scaler, scaled_data = normalize_df(processed_df)
    data, n_windows = rolling_window(processed_df, scaled_data, dataset)

    # get iid data
    real_series_iter, random_series = iid_converter(data, n_windows, dataset)

    # parameters
    hidden_dim = 24
    num_layers = 3
    n_seq = 26

    # 2. generate building block of Time GAN
    embedder = make_rnn(
        n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Embedder"
    )
    recovery = make_rnn(
        n_layers=3, hidden_units=hidden_dim, output_units=n_seq, name="Recovery"
    )
    generator = make_rnn(
        n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Generator"
    )

    discriminator = make_rnn(
        n_layers=3, hidden_units=hidden_dim, output_units=1, name="Discriminator"
    )

    supervisor = make_rnn(
        n_layers=2, hidden_units=hidden_dim, output_units=hidden_dim, name="Supervisor"
    )

    # training hyperparameter
    X, Z = get_input_placeholders()
    train_steps = 5  # 10000
    gamma = 1

    # loss function
    mse = MeanSquaredError()
    bce = BinaryCrossentropy()

    # 3. auto-encoder training
    H = embedder(X)
    X_tilde = recovery(H)
    autoencoder = Model(inputs=X, outputs=X_tilde, name="Autoencoder")
    plot_model(
        autoencoder,
        to_file=("../../imgs/autoencoder_" + dataset + ".png"),
        show_shapes=True,
    )

    # optimizer
    autoencoder_optimizer = Adam()

    # autoencoder supervised learning
    for step in tqdm(range(train_steps)):
        X_ = next(real_series_iter)
        step_e_loss_t0 = train_autoencoder_init(
            X_, autoencoder, embedder, recovery, autoencoder_optimizer, mse
        )
        with writer.as_default():
            tf.summary.scalar("Loss Autoencoder Init", step_e_loss_t0, step=step)

    # 4. joint training of generator and discriminator
    E_hat = generator(Z)
    H_hat = supervisor(E_hat)
    Y_fake = discriminator(H_hat)

    adversarial_supervised = Model(
        inputs=Z, outputs=Y_fake, name="AdversarialNetSupervised"
    )
    plot_model(
        adversarial_supervised,
        to_file=("../../imgs/adversarial_" + dataset + ".png"),
        show_shapes=True,
    )

    # adversarial architecture in latent space
    Y_fake_e = discriminator(E_hat)
    adversarial_emb = Model(inputs=Z, outputs=Y_fake_e, name="AdversarialNet")
    plot_model(
        adversarial_emb,
        to_file=("../../imgs/adversarial_emd_" + dataset + ".png"),
        show_shapes=True,
    )

    # mean and variance loss
    X_hat = recovery(H_hat)
    synthetic_data = Model(inputs=Z, outputs=X_hat, name="SyntheticData")
    plot_model(
        synthetic_data,
        to_file=("../../imgs/synthetic_data_" + dataset + ".png"),
        show_shapes=True,
    )

    # architecture for real data
    Y_real = discriminator(H)
    discriminator_model = Model(inputs=X, outputs=Y_real, name="DiscriminatorReal")
    plot_model(
        discriminator_model,
        to_file=("../../imgs/discriminator" + dataset + ".png"),
        show_shapes=True,
    )

    # optimizer
    generator_optimizer = Adam()
    discriminator_optimizer = Adam()
    embedding_optimizer = Adam()


if __name__ == "__main__":
    main()
