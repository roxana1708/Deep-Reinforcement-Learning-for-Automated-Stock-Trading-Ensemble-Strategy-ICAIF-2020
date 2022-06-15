# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
from model.saved_models import *
import os


import tensorflow

def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    # preprocessed_path = "done_data.csv" # OLD INDICATORS
    # preprocessed_path = "done_data_new_indicators_v12.csv" FINAL
    # preprocessed_path = "done_data_new_indicators_v14.csv" STOCHASTIC RSI
    # preprocessed_path = "done_data_new_indicators_v8.csv" TRIX+SMMA+CCI+PPO
    preprocessed_path = "done_data_new_indicators_v9.csv" # TRIX+SMMA+RSI+PPO
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        # data = pd.concat((x.query("index % 30 < 15") for x in data_read), ignore_index=True)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    # print(data.head())
    # print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    
    ## Ensemble Strategy
    # print(data[0:15:15])
    # print("LEN")
    # print(len(data))
    # data_param = []
    # print(2*len(data)/15)
    # for i in range(0, int(2*len(data)/15)):
    #     if i % 2 == 0:
    #         # print(data[(i*15):((i+1)*15)])
    #         # exit()
    #         # data_param = data_param + data[(i*15):((i+1)*15)]
    #         # print(data.keys())
    #         # exit()
    #         print(data[0:30])
    #     else:
    #         pass
    # print("New df")
    # print(data_param[0:30])

    run_ensemble_strategy(df=data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")


def run_our_saved_model() -> None:
    """ run training with saved model"""

    preprocessed_path = "done_data_new_indicators_v9.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    ## Ensemble Strategy
    run_ensemble_strategy_with_our_saved_model(df=data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    # _logger.info(f"saving model version: {_version}")


def run_original_saved_model() -> None:
    """ run training with saved model"""

    # read and preprocess data
    # preprocessed_path = "done_data.csv"
    # preprocessed_path = "done_data_new_indicators.csv"
    # preprocessed_path = "done_data_new_indicators_v2.csv"
    preprocessed_path = "done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        # data = pd.concat((x.query("index % 30 < 15") for x in data_read), ignore_index=True)
        # data = data_read
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    # print(data.head())
    # print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    ## Ensemble Strategy
    run_ensemble_strategy_with_original_saved_model(df=data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    # _logger.info(f"saving model version: {_version}")

# def plot_figures() -> None:


# print(tensorflow.__version__)
if __name__ == "__main__":
    # run_model()
    run_our_saved_model()
    # run_original_saved_model()
