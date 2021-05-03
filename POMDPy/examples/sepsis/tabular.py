import math
import random
import numpy as np
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.discrete_pomdp import DiscreteAction
from pomdpy.discrete_pomdp import DiscreteState
from pomdpy.discrete_pomdp import DiscreteObservation
from pomdpy.pomdp import HistoricalData
from pomdpy.pomdp import Model, StepResult
from pomdpy.util import console, config_parser

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
        done = bool(self.position == 0 or self.position == 10)
        next_position = self.model.make_next_position(self.position, action.bin_number)
        if not done:
            next_data.position = next_position
        return next_data

class BoxState(DiscreteState):
    def __init__(self, position):
        self.position = position
        self.terminal = bool(self.position == 0 or self.position == 10)
        
    def copy(self):
        return BoxState(self.position)

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

class TabularWorld():
    def __init__(self, missing=True, cost=-10):
        self.actions_n = 2 # L: 0, R: 1
        # 0 (Bad terminal) .... 5 ..... 10 (Good terminal), 11 (NA)
        self.states_n = 11
        # self.state = 0
        # self.obs = 0
        self.obs_n = self.states_n
        self.missing = missing
        if self.missing:
            self.na = 11
            self.obs_n += 1
            self.actions_n *= 2
        self.cost = cost
        self.max_steps = 15
        self.t = 0
        self.seed = random.seed()
    
        self.start = self.states_n // 2 # start state
        self.n_start_states = 2000
        self.ucb_coefficient = 3.0
        self.min_particle_count = 1000
        self.max_particle_count = 2000
        self.max_depth = 20
        self.action_selection_timeout = 60
        self.n_sims = 1000
        self.solver = 'POMCP'
        self.preferred_actions = True # not used
        self.test = 10
        self.epsilon_start = 0.9
        self.epsilon_minimum = 0.1
        self.epsilon_decay = 0.9
        self.discount = 0.98
        self.n_epochs = 100
        self.save = False
        self.timeout = 100000000
        self.initialize()

    def initialize(self):
        self.T = {}
        t0 = np.zeros((self.states_n, self.states_n))
        t1 = np.zeros((self.states_n, self.states_n))
        for i in range(1, self.states_n-1):
            p = 0.5 * 0.1 ** abs(i - 5)
            if i < 5:
                t0[i, i-1] = p
                t0[i, i+1] = 1 - p
                t1[i, i-1] = 1 - p
                t1[i, i+1] = p
            else:
                t0[i, i-1] = 1 - p
                t0[i, i+1] = p
                t1[i, i-1] = p
                t1[i, i+1] = 1 - p
        t0[0,0] = 1.
        t0[10,10] = 1.
        t1[0,0] = 1.
        t1[10,10] = 1.
        self.T[0] = t0
        self.T[1] = t1
        self.T[2] = t0
        self.T[3] = t1

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
        self.state = self.start
        self.obs = 5
        self.t = 0
        return self.obs

    '''
    rejection sampling
    '''
    def generate_particles(self, prev_belief, action, obs, n_particles, prev_particles):
        particles = []
        action_node = prev_belief.action_map.get_action_node(action)
        if action_node is None:
            return particles
        else:
            obs_map = action_node.observation_map
        child_node = obs_map.get_belief(obs)

        while particles.__len__() < n_particles:
            state = random.choice(prev_particles)
            result, is_legal = self.generate_step(state, action)
            if obs_map.get_belief(result.observation) is child_node:
                particles.append(result.next_state)
        return particles

    def make_next_state(self, state, action):
        if state.terminal:
            return state.copy(), False
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        probs = self.T[action][state,:]
        next_pos = np.random.choice(np.arange(self.states_n), 1, p=probs)[0]
        return BoxState(int(next_pos)), True

    def make_next_position(self, state, action):
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        probs = self.T[action][state,:]
        return int(np.random.choice(np.arange(self.states_n), 1, p=probs)[0])

    def get_all_observations(self):
        obs = {}
        for i in range(self.obs_n):
            obs[i] = i

    def make_observation(self, action, next_state):
        if action.bin_number >= 2 :
            obs_n = self.na # not observe
        else:
            obs_n = next_state.position
        return BoxObservation(obs_n)
        
    def make_reward(self, state, action, next_state, is_legal):
        rew = 0
        if not is_legal:
            return rew
        if action.bin_number < 2:
            rew = rew + self.cost
        if next_state.terminal:
            if next_state.position == 0:
                rew -= 100
            else:
                rew += 100
        return rew

    def generate_step(self, state, action):
        if type(action) is int:
            action = BoxAction(action)
        result = StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state, is_legal)
        result.is_terminal = result.next_state.terminal
        return result, is_legal
    
    def reset_for_simulation(self):
        pass

    def sample_an_init_state(self):
        return BoxState(self.start)

    def create_root_historical_data(self, solver):
        return PositionData(self, self.start, solver)

    def belief_update(self, old_belief, action, observation):
        pass

    def get_max_undiscounted_return(self):
        return 100

    def reset_for_epoch(self):
        self.t = 0
        # obs, rew, done, info = self.reset()

    def step(self, action):
        assert(self.state > 0 and self.state < self.states_n - 1)
        obs = True
        rew = 0
        if self.missing:
            if action > 2:
                obs = False
            action = action % 2
            if obs:
                rew += self.cost

        ### move to next state ###
        pr = 0.5 * 0.15 ** abs(self.state-self.start)
        acts = [-1, 1]
        if action == 0 and self.state > self.start or action == 1 and self.state < self.start:
            a = np.random.choice(acts, 1, p=[pr, 1-pr])[0]
        else: # action == 0 and self.state <5 or action == 1 and self.state > 5
            a = np.random.choice(acts, 1, p=[1-pr, pr])[0]
        self.state += a
        #########################

        done = bool(self.t == self.max_steps or
                self.state == 0 or self.state == self.states_n - 1)
        
        if self.state == 0:
            rew -= 100
        if self.state == self.states_n - 1:
            rew += 100
        self.t += 1
        if not obs:
            self.obs = self.na
        else:
            rew += self.cost
            self.obs = self.state
        
        return self.obs, rew, done, {'true': self.state}

    def render(self):
        pass
