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

def preprocess_df(dataset='batadal'):
    df = None
    tickers = None
    path = '../../../data/batadal_dataset03.csv'
    # Batadal dataset
    if(dataset=='batadal'):
        df = pd.read_csv(path)
        print(f"testing {df}")
        
        # Get only float values
        df = df[['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
         'F_PU1', 'F_PU2', 'F_PU4', 'F_PU7', 'F_PU8', 'F_PU10',
         'F_V2',
         'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 
         'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']]
        
        # Get attribute name
        tickers = list(df.columns)

    # SWaT dataset
    elif(dataset=='swat'):
        path=='data/swat_dataset_normal_v1.csv'
        df = pd.read_csv(path)
        
        # Get only float values
        df = df[['LIT101',
         'AIT201', 'AIT202', 'AIT203',
         'DPIT301', 'FIT301', 'LIT301',
         'AIT402', 'LIT401',
         'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT503',
         'PIT501', 'PIT503', 'FIT601']]
                # Get attribute name
                
        tickers = list(df.columns)
    else:
        print("Can't find path")
        return

    return df, tickers

def plot_df(processed_df, tickers):
    axes = processed_df.div(processed_df.iloc[0]).plot(subplots=True,
                                   figsize=(14, 26),
                                   layout=(13,2),
                                   title=tickers,
                                   legend=False,
                                   rot=0,
                                   lw=1,
                                   color='k')
    for ax in axes.flatten():
        ax.set_xlabel("")
        
    plt.suptitle("Normalized Time-Series")
    plt.gcf().tight_layout()
    sns.despine();
    plt.show()      
    
def corr_plot(processed_df):
    sns.clustermap(processed_df.corr(),
              annot=True,
              fmt='0.2f',
              cmap=sns.diverging_palette(h_neg=20,
                                         h_pos=220), center=0)
    plt.show()
    
def normalize_df(processed_df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(processed_df).astype(np.float32)
    return scaler, scaled_data

def rolling_window(processed_df, scaled_data):
    """Creates rolling window sequence and convert ts data to iid
    """
    data = []
    for i in range(len(processed_df) - seq_len):
        data.append(scaled_data[i:i + seq_len])
    n_windows = len(data)

    return data, n_windows

def iid_converter(data, n_windows, dataset='batadal'):
    # Parameters
    seq_len = None
    n_seq = 26
    batch_size = 128

    if dataset=='batadal':
        seq_len = 24
    elif dataset=='swat':
        seq_len = 18
    else:
        print("Please check your 'dataset' hyperparameter again.")
        return

    def make_random_data():
        while True:
            np.random.uniform(low=0, high=1, size=(seq_len, n_seq))
    
    
    # Convert real ts data to iid dataset
    real_series = (tf.data.Dataset
                .from_tensor_slices(data)
                .shuffle(buffer_size=n_windows)
                .batch(batch_size))
    
    real_series_iter = iter(real_series.repeat())
    
    random_series = iter(tf.data.Dataset
                        .from_generator(make_random_data, output_types=tf.float32)
                        .batch(batch_size)
                        .repeat())
    
    return real_series_iter, random_series