from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import sys
sys.path.append('acno_mdp/locf/env/sepsisSimDiabetes')
from sepsis_tabular import SepsisEnv
import pandas as pd
# from env_Tmaze import EnvTMaze
import numpy as np
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INIT_STATE = 256

"""
Generic DRQN with LSTM:
Start from a fixed patient state represented by 256.
One hot encode observations, including the missingness observation
and learn the LSTM policy mapping from observations.
"""

class ReplayMemory(object):
    def __init__(self, max_epi_num=50, max_epi_len=300):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=self.max_epi_num)
        self.is_av = False
        self.current_epi = 0
        self.memory.append([])

    def reset(self):
        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def remember(self, state, action, reward):
        if len(self.memory[self.current_epi]) < self.max_epi_len:
            self.memory[self.current_epi].append([state, action, reward])

    # samples a trajectory of length 5
    def sample(self):
        epi_index = random.randint(0, len(self.memory)-2)
        if self.is_available():
            return self.memory[epi_index]
        else:
            return []

    def size(self):
        return len(self.memory)

    def is_available(self):
        self.is_av = True
        if len(self.memory) <= 1:
            self.is_av = False
        return self.is_av

    def print_info(self):
        for i in range(len(self.memory)):
            print('epi', i, 'length', len(self.memory[i]))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DRQN(nn.Module):
    def __init__(self, N_action):
        super(DRQN, self).__init__()
        self.lstm_i_dim = 16    # input dimension of LSTM
        self.lstm_h_dim = 16     # output dimension of LSTM
        self.lstm_N_layer = 1   # number of layers of LSTM
        self.N_action = N_action
        self.flat1 = nn.Linear(721, self.lstm_h_dim)
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(self.lstm_h_dim, 16)
        self.fc2 = nn.Linear(16, self.N_action)

    def forward(self, x, hidden):
        h2 = self.flat1(x)
        h2 = h2.unsqueeze(0)
        h3, new_hidden = self.lstm(h2, hidden)
        h4 = F.relu(self.fc1(h3))
        h5 = self.fc2(h4)
        return h5, new_hidden

class Agent(object):
    def __init__(self, N_action, max_epi_num=50, max_epi_len=300):
        self.N_action = N_action
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.drqn = DRQN(self.N_action)
        self.buffer = ReplayMemory(max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len)
        self.gamma = 0.7
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.drqn.parameters(), lr=1e-3)

    def remember(self, state, action, reward):
        self.buffer.remember(state, action, reward)


    def train(self):
        if self.buffer.is_available():
            memo = self.buffer.sample()
            obs_list = []
            action_list = []
            reward_list = []

            # batch_size, traj len, state dim
            # hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
            Q_est = []
            Qs = []
            hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
            for i in range(len(memo)):
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
                obs = torch.Tensor(memo[i][0]).unsqueeze(0)
                Q, hidden = self.drqn.forward(obs, hidden)
                Qs.append(Q)
                Q_est.append(Q.clone())

            losses = []
            for t in range(len(memo) - 1):
                max_next_q = torch.max(Q_est[t+1]).clone().detach()
                q_target = Qs[t].clone().detach()
                q_target[0, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
                losses.append(self.loss_fn(Qs[t], q_target))

            T = len(memo) - 1
            q_target = Qs[T].clone().detach()
            q_target[0, 0, action_list[T]] = reward_list[T]
            losses.append(self.loss_fn(Qs[T], q_target))
            loss = torch.stack(losses).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, obs, hidden, epsilon):
        obs = torch.Tensor(obs)
        if len(obs.shape) == 1: # batch sz: 1
            obs = obs.unsqueeze(0)
        if random.random() > epsilon:
            q, new_hidden = self.drqn.forward(obs, hidden)
            action = q[0].max(1)[1].data[0].item()
        else:
            q, new_hidden = self.drqn.forward(obs, hidden)
            action = random.randint(0, self.N_action-1)
        return action, new_hidden

def get_decay(epi_iter):
    decay = math.pow(0.999, epi_iter)
    if decay < 0.05:
        decay = 0.05

    # if epi_iter >= 1000:
    #    decay = 0.0

    return decay

def action_encod(action):
    dictionary = {0: '0A_0E_0V',
                1: '0A_0E_1V',
                2: '0A_1E_0V',
                3: '0A_1E_1V',
                4: '1A_0E_0V',
                5: '1A_0E_1V',
                6: '1A_1E_0V',
                7: '1A_1E_1V'}
    return dictionary[action]

def one_hot_encoding(obs):
    # return [obs]
    one_hot = np.zeros((721))
    one_hot[obs] = 1
    return one_hot

if __name__ == '__main__':
    random.seed()
    import time
    start = time.time()
    obs_cost = -0.05
    env = SepsisEnv(obs_cost=obs_cost, no_missingness=False)
    max_epi_iter = 2000
    max_MC_iter = 5
    agent = Agent(N_action=16, max_epi_num=1000, max_epi_len=max_MC_iter)
    train_curve = []
    for epi_iter in range(max_epi_iter):
        random.seed()
        obs = env.reset(INIT_STATE)
        obs = one_hot_encoding(obs)
        hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
        returns = 0
        for MC_iter in range(max_MC_iter):
            # env.render()
            action, hidden = agent.get_action(obs, hidden, get_decay(epi_iter))
            obs, reward, done, info = env.step(action)
            obs = one_hot_encoding(obs)
            returns += reward * agent.gamma ** (MC_iter)
            agent.remember(obs, action, reward)
            # if reward != 0 or MC_iter == max_MC_iter-1:
            if done or MC_iter >= max_MC_iter-1:
                agent.buffer.create_new_epi()
                break
        print('Episode', epi_iter, 'returns', returns) #  'where', env.if_up)
        if epi_iter % 1 == 0:
            train_curve.append(returns)
        if agent.buffer.is_available():
            agent.train()

    # STOP TRAINING and only use the last model to evaluate on 100 new trials
    return_list = []
    for eval_iter in range(50):
        obs = env.reset(INIT_STATE)
        hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
        returns = 0
        obs = one_hot_encoding(obs)
        for MC_iter in range(max_MC_iter):
            # env.render()
            action, hidden = agent.get_action(obs, hidden, 0)
            obs, reward, done, info = env.step(action)
            observed = bool(action < 8)
            obs = one_hot_encoding(obs)
            print('Observed:', observed)
            print('Action: ', action_encod(action % 8))
            returns += reward * agent.gamma ** (MC_iter)
            agent.remember(obs, action, reward)
            # if reward != 0 or MC_iter == max_MC_iter-1:
            if done or MC_iter == max_MC_iter-1:
                agent.buffer.create_new_epi()
                break
        print('Returns: ', returns)

        return_list.append(returns)
    print('Obs cost: ', obs_cost)
    print('Eval mean return: ', np.mean(return_list))
    print('Eval ste return: ', np.std(return_list))
    print('Eval ste return: ', np.std(return_list)/ np.sqrt(50))

    # new is same as w/out "new" but just want to make sure the stats are consistent
    # np.save("exp_0523/drqn/new_2k_hidden_DRQN_sepsis_obs_cost_0.05_test.npy", np.array(train_curve))
    print("drqn runtime: ", time.time() - start)
