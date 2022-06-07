# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
# from stable_baselines import GAIL, SAC
# from stable_baselines import ACER
# from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import SAC
# from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines import TRPO

# from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

import tensorflow
# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    print(len(test_data.index.unique()))
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy_with_saved_model(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    sac_sharpe_list = []
    td3_sharpe_list = []
    trpo_sharpe_list = []

    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        # print(train)
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        # print(validation)
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        # print("OBS val")
        # print(len(obs_val[0]))
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        # print("======A2C Training========")
        # model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30000)
        # print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_a2c = get_validation_sharpe(i)
        # print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======SAC Training========")
        # model_sac = train_SAC(env_train, model_name="SAC_30k_dow_{}".format(i), timesteps=30000)
        model_sac = SAC.load("trained_models/newIndicators_30stocks/SAC_30k_dow_945")
                             # .format(i))
        print("======SAC Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_sac, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_sac = get_validation_sharpe(i)
        print("SAC Sharpe Ratio: ", sharpe_sac)

        print("======TD3 Training========")
        # model_td3 = train_TD3(env_train, model_name="TD3_30k_dow_{}".format(i), timesteps=30000)
        model_td3 = TD3.load("trained_models/newIndicators_30stocks/TD3_30k_dow_945")
        # .format(i))
        print("======TD3 Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_td3 = get_validation_sharpe(i)
        print("TD3 Sharpe Ratio: ", sharpe_td3)

        print("======TRPO Training========")
        # model_trpo = train_TRPO(env_train, model_name="TRPO_40k_dow_{}".format(i), timesteps=40000)
        model_trpo = TRPO.load("trained_models/newIndicators_30stocks/TRPO_40k_dow_945")
                               # .format(i))
        print("======TRPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_trpo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_trpo = get_validation_sharpe(i)
        print("TRPO Sharpe Ratio: ", sharpe_trpo)

        # print("======PPO Training========")
        # model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        # print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_ppo = get_validation_sharpe(i)
        # print("PPO Sharpe Ratio: ", sharpe_ppo)

        # print("======DDPG Training========")
        # model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=10000)
        # #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        # print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_ddpg = get_validation_sharpe(i)

        # ppo_sharpe_list.append(sharpe_ppo)
        # a2c_sharpe_list.append(sharpe_a2c)
        # ddpg_sharpe_list.append(sharpe_ddpg)
        sac_sharpe_list.append(sharpe_sac)
        td3_sharpe_list.append(sharpe_td3)
        trpo_sharpe_list.append(sharpe_trpo)

        # Model Selection based on sharpe ratio
        if (sharpe_sac >= sharpe_td3) & (sharpe_sac >= sharpe_trpo):
            model_ensemble = model_sac
            model_use.append('SAC')
        elif (sharpe_td3 >= sharpe_sac) & (sharpe_td3 >= sharpe_trpo):
            model_ensemble = model_td3
            model_use.append('TD3')
        else:
            model_ensemble = model_trpo
            model_use.append('TRPO')

        # if (sharpe_td3 <= sharpe_sac):
        #     model_ensemble = model_sac
        #     model_use.append('SAC')
        # else:
        #     model_ensemble = model_td3
        #     model_use.append('TD3')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)

        print("============Trading Done============")
        print(last_state_ensemble)
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
    print(model_use)

