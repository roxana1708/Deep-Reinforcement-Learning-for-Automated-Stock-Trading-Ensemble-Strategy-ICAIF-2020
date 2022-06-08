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

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_SAC(env_train, model_name, timesteps=25000):
    """SAC model"""

    start = time.time()
    model = SAC('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (SAC): ', (end - start) / 60, ' minutes')
    return model


def train_TD3(env_train, model_name, timesteps=25000):
    """TD3 model"""

    start = time.time()
    model = TD3('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (TD3): ', (end - start) / 60, ' minutes')
    return model


def train_TRPO(env_train, model_name, timesteps=25000):
    """TRPO model"""

    start = time.time()
    model = TRPO('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (TRPO): ', (end - start) / 60, ' minutes')
    return model
# def train_ACER(env_train, model_name, timesteps=25000):
#     start = time.time()
#     model = ACER('MlpPolicy', env_train, verbose=0)
#     model.learn(total_timesteps=timesteps)
#     end = time.time()
#
#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (A2C): ', (end - start) / 60, ' minutes')
#     return model
#
#
# def train_DDPG(env_train, model_name, timesteps=10000):
#     """DDPG model"""
#
#     # add the noise objects for DDPG
#     n_actions = env_train.action_space.shape[-1]
#     param_noise = None
#     action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
#
#     start = time.time()
#     model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
#     model.learn(total_timesteps=timesteps)
#     end = time.time()
#
#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (DDPG): ', (end-start)/60,' minutes')
#     return model
#
# def train_PPO(env_train, model_name, timesteps=50000):
#     """PPO model"""
#
#     start = time.time()
#     model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
#     #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)
#
#     model.learn(total_timesteps=timesteps)
#     end = time.time()
#
#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (PPO): ', (end - start) / 60, ' minutes')
#     return model
#
# def train_GAIL(env_train, model_name, timesteps=1000):
#     """GAIL Model"""
#     #from stable_baselines.gail import ExportDataset, generate_expert_traj
#     start = time.time()
#     # generate expert trajectories
#     model = SAC('MLpPolicy', env_train, verbose=1)
#     generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)
#
#     # Load dataset
#     dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
#     model = GAIL('MLpPolicy', env_train, dataset, verbose=1)
#
#     model.learn(total_timesteps=1000)
#     end = time.time()
#
#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (PPO): ', (end - start) / 60, ' minutes')
#     return model


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
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    if df_total_value['daily_return'].std() != 0:
        sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    else:
        sharpe = 0

    return sharpe


def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    # test_ex = df[::30]
    # print(test_ex[0:30])
    # exit()
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
    model_sac = []
    model_td3 = []
    model_trpo = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    print("TURBULANCE INSAMPLE DATA")
    insample_turbulence = df[(df.datadate < 20151000) & (df.datadate >= 20090000)]
    print(insample_turbulence)
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    print(insample_turbulence)
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
    print(insample_turbulence_threshold)

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
        print(train)
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])

        print("======SAC Training========")
        # if model_sac:
        #     print("update training")
        #
        #     start = time.time()
        #     model_sac.learn(total_timesteps=30000)
        #     end = time.time()
        #     model_name = "SAC_30k_dow"
        #     model_sac.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        #     print('Training time (SAC): ', (end - start) / 60, ' minutes')
        # else:
        #     print("first training")
        model_sac = train_SAC(env_train, model_name="SAC_25k_dow_{}".format(i), timesteps=25000)
        print("======SAC Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_sac, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_sac = get_validation_sharpe(i)
        print("SAC Sharpe Ratio: ", sharpe_sac)

        print("======TD3 Training========")
        # if model_td3:
        #     print("update training")
        #     start = time.time()
        #     model_td3.learn(total_timesteps=20000)
        #     end = time.time()
        #     model_name = "TD3_30k_dow"
        #     model_td3.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        #     print('Training time (TD3): ', (end - start) / 60, ' minutes')
        #
        # else:
        #     print("first training")
        model_td3 = train_TD3(env_train, model_name="TD3_35k_dow_{}".format(i), timesteps=35000)
        print("======TD3 Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_td3 = get_validation_sharpe(i)
        print("TD3 Sharpe Ratio: ", sharpe_td3)
        #
        # print("======TRPO Training========")
        # if model_trpo:
        #     print("update training")
        #     start = time.time()
        #     model_trpo.learn(total_timesteps=20000)
        #     end = time.time()
        #     model_name = "TRPO_30k_dow"
        #     model_trpo.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        #     print('Training time (TRPO): ', (end - start) / 60, ' minutes')
        # else:
        #     print("first training")
        model_trpo = train_TRPO(env_train, model_name="TRPO_40k_dow_{}".format(i), timesteps=40000)

        print("======TRPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_trpo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_trpo = get_validation_sharpe(i)
        print("TRPO Sharpe Ratio: ", sharpe_trpo)

        sac_sharpe_list.append(sharpe_sac)
        td3_sharpe_list.append(sharpe_td3)
        trpo_sharpe_list.append(sharpe_trpo)

        # Model Selection based on sharpe ratio

        if (sharpe_sac >= sharpe_td3) & (sharpe_sac >= sharpe_trpo):
            model_ensemble = model_sac
            model_use.append('SAC')
        elif (sharpe_td3 > sharpe_sac) & (sharpe_td3 > sharpe_trpo):
            model_ensemble = model_td3
            model_use.append('TD3')
        else:
            model_ensemble = model_trpo
            model_use.append('TRPO')

        # model_ensemble = model_sac
        # model_use.append('SAC')
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
    # print(sac_sharpe_list)
    # print(td3_sharpe_list)
    # print(trpo_sharpe_list)
