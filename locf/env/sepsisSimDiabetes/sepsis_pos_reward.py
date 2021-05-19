import numpy as np, random
from MDP_pos_reward import MDP
from State import State
from Action import Action

from gym import spaces
import gym

CONSTANT = 0.25

class SepsisEnv(gym.Env):
    def __init__(self,
                obs_cost=0., # keep it at zero cost (since obs cost added in pomdpy/sepsis code)
                init_state=96,
                noise=False,
                per_step_reward=False,
                counter=False,
                locf=False,
                no_missingness=False,
                action_aug=False,
                ):
        self.timestep = 0
        self.max_t = 5
        self.env = None
        self.locf = locf
        self.cost = obs_cost
        self.counter = counter
        self.viewer = None
        actions_n = 8
        states_n = 720 # addtional one for missingness
        obs_n = states_n 
        if not no_missingness:
            actions_n *= 2
            obs_n += 1
        self.action_space = spaces.Discrete(actions_n)
        self.observation_space = spaces.Discrete(obs_n)
        self.state_space = spaces.Discrete(states_n)
        self.state = None
        self.obs = None
        self.reset(init_state)

    # any action idx between 0~7 is observing
    def separate_action(self, action):
        return (action % 8), bool(action < 8)

    def step(self, action, prev_state=None):
        a, obs = self.separate_action(action)
        self.timestep += 1
        if prev_state is not None: # fix prev state by creating a new env
            self.env = MDP(init_state_idx=prev_state, init_state_idx_type='obs', p_diabetes=0.)
        reward = self.env.transition(Action(action_idx=a))
        state = self.env.state.get_state_idx()
        # Add +1 to every vital value since 0 is used for NULL
        done = bool(self.timestep == self.max_t or reward != CONSTANT)
        if obs:
            self.obs = state
            reward += self.cost
        else:
            self.obs = self.observation_space.n
        return self.obs, reward, done, {'true_state': state}

    def reset(self, init_idx):
        self.timestep = 0
        self.env = MDP(init_state_idx=init_idx, 
                        init_state_idx_type='obs', 
                        p_diabetes=0.)
        state = self.env.state.get_state_idx()
        # Add +1 to every vital value since 0 is used for NULL
        self.state = state
        self.obs = state
        return self.obs

        def render(self, mode='human'):
            pass
