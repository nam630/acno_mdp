import math
import random
import numpy as np
import sys
sys.path.append('/next/u/hjnam/locf/env/sepsisSimDiabetes')
sys.path.append('/next/u/hjnam/POMDPy')
import pickle
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.discrete_pomdp import DiscreteAction
from pomdpy.discrete_pomdp import DiscreteState
from pomdpy.discrete_pomdp import DiscreteObservation
from pomdpy.pomdp import HistoricalData
from pomdpy.pomdp import Model, StepResult
from pomdpy.util import console, config_parser

from sepsis_tabular import SepsisEnv

class PositionData(HistoricalData):
    def __init__(self, model, position, solver):
        self.model = model
        self.solver = solver
        self.position = position
        self.legal_actions = self.generate_legal_actions 

    def generate_legal_actions(self):
        legal_actions = []
        for i in range(self.model.actions_n):
            legal_actions.append(i)
        return legal_actions

    def shallow_copy(self):
        return PositionData(self.model, self.position, self.solver)
    
    def copy(self):
        return self.shallow_copy()

    def update(self, other_belief):
        pass

    def create_child(self, action, observation):
        next_data = self.copy() # deep/shallow copy
        next_position, is_legal = self.model.make_next_position(self.position, action.bin_number)
        next_data.position = next_position
        return next_data

class BoxState(DiscreteState):
    def __init__(self, position, is_terminal=False, r=None):
        self.position = position
        self.terminal = is_terminal
        self.final_rew = r

    def copy(self):
        return BoxState(self.position, 
                        is_terminal=self.terminal, 
                        r=self.final_rew)
        
    def to_string(self):
        return str(self.position)

    def print_state(self):
        pass

    def as_list(self):
        pass

    def distance_to(self):
        pass

class BoxAction(DiscreteAction):
    def __init__(self, bin_number):
        self.bin_number = bin_number

    def print_action(self):
        pass

    def to_string(self):
        return str(self.bin_number)

    def distance_to(self):
        pass

    def copy(self):
        return BoxAction(self.bin_number)

class BoxObservation(DiscreteObservation):
    def __init__(self, position):
        self.position = position
        self.bin_number = position

    def copy(self):
        return BoxObservation(self.position)

    def to_string(self):
        return str(self.position)

    def print_observation(self):
        pass

    def as_list(self):
        pass

    def distance_to(self):
        pass

class Sepsis():
    def __init__(self, init_state_idx, missing=True, cost=-0, is_mdp=0):
        self.init_state = init_state_idx
        self.sim = SepsisEnv(init_state=self.init_state, no_missingness=bool(is_mdp == 1)) 
        self.actions_n = self.sim.action_space.n
        self.states_n = self.sim.state_space.n
        self.obs_n = self.sim.observation_space.n
        self.missing = bool(is_mdp==0)
        if self.missing:
            self.na = self.sim.state_space.n
        self.cost = cost
        self.max_steps = 5 
        self.t = 0
        self.seed = random.seed() 
        self.n_start_states = 2000
        self.ucb_coefficient = 3.0
        self.min_particle_count = 1000
        self.max_particle_count = 2000
        self.max_depth = 5
        self.action_selection_timeout = 60
        self.n_sims = 1000
        if is_mdp == 0:
            self.solver = 'POMCP'
        else:
            self.solver = 'MCP'
        print('Solving with ', self.solver)
        self.preferred_actions = True # not used
        self.test = 10
        self.epsilon_start = 0.9
        self.epsilon_minimum = 0.1
        self.epsilon_decay = 0.9
        self.discount = 0.98
        self.n_epochs = 100
        self.save = False
        self.timeout = 10000
        
        ##### Load empirical model #####
        self.empi_model = pickle.load(open('/next/u/hjnam/locf/sepsis/0411/p_256_/256model_pi.obj','rb'))
        ################################

    def update(self, step_result):
        pass

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def get_all_actions(self):
        all_actions = []
        for i in range(self.actions_n):
            all_actions.append(BoxAction(i))
        return all_actions

    def get_legal_actions(self, state):
        all_actions = []
        for i in range(self.actions_n):
            all_actions.append(i)
        return all_actions

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def reset(self):
        self.t = 0

    '''
    rejection sampling
    '''
    def generate_particles(self, prev_belief, action, obs, n_particles, prev_particles, mdp):
        particles = []
        action_node = prev_belief.action_map.get_action_node(action)
        if action_node is None:
            return particles
        else:
            obs_map = action_node.observation_map
        child_node = obs_map.get_belief(obs)

        while particles.__len__() < n_particles:
            state = random.choice(prev_particles)
            if mdp:
                result, is_legal = self.generate_step(state, action, is_mdp=True)
            else:
                result, is_legal = self.generate_step(state, action)
            if obs_map.get_belief(result.observation) is child_node:
                particles.append(result.next_state)
        return particles

    '''
    Can use a plug-in empirical simulator
    Inp: action (dtype int), state (dtype int)
    1) # of samples
    2) exact noise level
    '''
    def empirical_simulate(self, state, action):
        # return self.sim.step(action, state)
        # rew = 0
        # if action > 7:
        #    rew += self.cost
        action = action % 8
        if (state, action) in self.empi_model.keys():
            p = self.empi_model[(state, action)]
            p /= sum(p)
            state = np.random.choice(720, 1, p=p)[0] # sample according to probs
        else:
            state = np.random.randint(0, 720, 1)[0] # random sample
        # use true environment for reward estimation
        temp = SepsisEnv(init_state=state)
        temp.env.state.diabetic_idx = 0 # always non-diabetic
        temp.env.state.set_state_by_idx(int(state), idx_type='obs', diabetic_idx = int(0))
        # only add env reward (x observation cost)
        rew = temp.env.calculateReward()
        return BoxState(int(state), is_terminal=bool(rew != 0), r=rew), True

    # def make_next_state(self, state, action):
    #    if state.terminal:
    #        return state.copy(), False
    #    if type(action) is not int:
    #        action = action.bin_number
    #    if type(state) is not int:
    #        state = state.position
    #    # this should be an imagined step in the learned simulator
    #    _, rew, done, info = self.empirical_simulate(action, state)
    #    next_pos = info['true_state'] 
    #    return BoxState(int(next_pos), is_terminal=done, r=rew), True

    '''
    In the real env, observation = state \cup NA
    but always return TRUE
    '''
    def take_real_state(self, state, action):
        if state.terminal:
            return state.copy(), False
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        _, rew, done, info = self.sim.step(action, state)
        # if action < self.actions_n // 2 :
        state = info['true_state'] 
        return BoxState(int(state), is_terminal=done, r=rew), True

    '''
    Should always return the predicted/underlying state
    '''
    def make_next_position(self, state, action):
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        return self.empirical_simulate(state, action)
    #    # should be through the learned simulator
    #    _, _, _, info = self.empirical_simulate(action, state)
    #    next_position = info['true_state']
    #    return int(next_position)

    def get_all_observations(self):
        obs = {}
        for i in range(self.obs_n):
            obs[i] = i

    '''
    Should return observation based on action
    '''
    def make_observation(self, action, next_state, always_obs=False):
        if action.bin_number < self.actions_n // 2 or always_obs:
            obs = next_state.position # not observe
        else:
            obs = self.na
        return BoxObservation(obs)
        
    def make_reward(self, state, action, next_state, is_legal, always_obs=False):
        rew = 0
        if not is_legal:
            return rew
        if action.bin_number < self.actions_n // 2 or always_obs:
            rew = rew + self.cost
        # if next_state.terminal:
        rew += next_state.final_rew
        return rew

    def generate_step(self, state, action, _true=False, is_mdp=False):
        if type(action) is int:
            action = BoxAction(action)
        result = StepResult()
        # Based on the simulator, next_state is true next state or imagined by the simulator
        if _true:
            result.next_state, is_legal = self.take_real_state(state, action)
        # Use true simulator for taking actions (only for eval)
        else:
            result.next_state, is_legal = self.make_next_position(state, action)
        
        ###  for true runs #####
        # result.next_state, is_legal = self.take_real_state(state, action)
        ########################
        
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state, always_obs=is_mdp)
        result.reward = self.make_reward(state, action, result.next_state, is_legal, always_obs=is_mdp)
        result.is_terminal = result.next_state.terminal
        return result, is_legal

    
    '''
    def mdp_generate_step(self, state, action):
        if type(action) is int:
            action = BoxAction(action)
        result = StepResult()
        result.next_state, is_legal = self.make_next_position(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state, always_obs=True)
        result.reward = self.make_reward(state, action, result.next_state, is_legal, always_obs=True)
        result.is_terminal = result.next_state.terminal
        return result, is_legal
    '''

    def reset_for_simulation(self):
        pass

    def sample_an_init_state(self):
        return BoxState(self.init_state)

    def create_root_historical_data(self, solver):
        return PositionData(self, self.init_state, solver)

    def belief_update(self, old_belief, action, observation):
        pass

    def get_max_undiscounted_return(self):
        return 1

    def reset_for_epoch(self):
        self.t = 0

    def render(self):
        pass
