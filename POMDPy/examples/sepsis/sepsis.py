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
import time
from sepsis_pos_reward import SepsisEnv

debug_mode = False

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

    def __eq__(self, other_state):
        return self.position == other_state.position

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

    def __eq(self, other_action):
        return self.bin_number == other_action.bin_number

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

    def __eq__(self, other_obs):
        return self.position == other_obs.position

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
        self.n_start_states = 5000
        self.ucb_coefficient = 3.0
        self.min_particle_count = 3000
        self.max_particle_count = 5000
        self.max_depth = 5
        self.action_selection_timeout = 60
        self.particle_selection_timeout = 0.2
        self.n_sims = 1000000 # all new runs with this 10000 # 10000000
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
        self.discount = 0.7
        self.n_epochs = 25000
        self.save = False
        self.timeout = 7200000
        
        # only for debugging
        self.real_state = 256
        if debug_mode:
            self.empi_model = pickle.load(open('/next/u/hjnam/locf/sepsis/0411/p_256_/256model_pi.obj','rb'))
        else: 
            # starts with an empty model 
            self.t_estimates = np.zeros((720, 8, 720))
            self.r_estimates = np.zeros((720, 8))
            # use a prior of 1 for all (s, a, s') 
            self.n_counts = np.ones((720, 8)) * 720
            self.r_counts = np.zeros((720, 8))
            self.t_counts = np.ones((720, 8, 720))

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
    Use rejection sampling to generate a new set of belief particles (from prev_particles to particles)
    '''
    def generate_particles(self, prev_belief, action, obs, n_particles, prev_particles, mdp):
        particles = []
        action_node = prev_belief.action_map.get_action_node(action)
        if action_node is None:
            return particles
        else:
            obs_map = action_node.observation_map
        child_node = obs_map.get_belief(obs)
        start = time.time()

        print('REJECTION SAMPLING STARTED, {}'.format(obs.position))
        
        while particles.__len__() < n_particles:
            state = random.choice(prev_particles)
            if mdp:
                result, is_legal = self.generate_step(state, action, is_mdp=True)
            else:
                result, is_legal = self.generate_step(state, action)
            # if null (i.e., 720) obs, any state particle CAN be added
            if result.observation.position == 720 or result.observation == obs: 
                assert(result.next_state.position < 720)
                particles.append(result.next_state)
                if particles.__len__() % 500 == 0: # logging for debugging
                    print(particles.__len__(), time.time() - start)
            if time.time() - start > self.particle_selection_timeout:
                if obs.position != 720:
                    print('prev pos:', state.position, particles.__len__(), 'prob:', self.t_estimates[state.position, action.bin_number % 8, obs.position])
                    print('real pos{}, real transition:'.format(self.real_state), self.t_estimates[self.real_state, action.bin_number, obs.position])
                if particles.__len__() > 3: # 3 too long?
                    print('REJECTION timeout:', time.time() - start)
                    break
        
        while particles.__len__() < n_particles:
            state = random.choice(particles)
            new_state = state.copy()
            particles.append(new_state)

        return particles

    '''
    Can use a plug-in empirical simulator
    Inp: action (dtype int), state (dtype int)
    1) # of samples
    2) exact noise level
    '''
    def empirical_simulate(self, state, action):
        action = action % 8
        if debug_mode:
            if (state, action) in self.empi_model.keys():
                p = self.empi_model[(state, action)]
                p /= sum(p)
                next_state = np.random.choice(720, 1, p=p)[0] # sample according to probs
            else:
                next_state = np.random.randint(0, 720, 1)[0] # random sample
        else:
            p = self.t_estimates[state,action,:]
            next_state = np.random.choice(720, 1, p=p)[0] # sample according to probs
            rew = self.r_estimates[state, action]
        
        # use true environment for reward estimation
        temp = SepsisEnv(init_state=next_state)
        temp.env.state.diabetic_idx = 0 # always non-diabetic
        temp.env.state.set_state_by_idx(int(next_state), idx_type='obs', diabetic_idx = int(0))
        terminal = temp.env.state.check_absorbing_state()
        if debug_mode:
            # only add env reward (x observation cost)
            rew = temp.env.calculateReward()
        return BoxState(int(next_state), is_terminal=terminal, r=rew), True


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
        temp = SepsisEnv(init_state=state) # next_state)
        temp.env.state.set_state_by_idx(state, idx_type='obs', diabetic_idx=0)
        next_state, rew, done, info = temp.step(action % 8, state)
        # print("next obs:", next_state, "state: ", state, "last real state: ", self.real_state)
        # print("true prob: ", self.t_estimates[self.real_state, action % 8, info['true_state']])
        # print(np.argwhere(self.t_estimates[self.real_state, action % 8, :] > 0.))
        state = info['true_state'] 
        self.real_state = state
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
        rew += next_state.final_rew
        return rew

    def generate_step(self, state, action, _true=False, is_mdp=False):
        if type(action) is int:
            action = BoxAction(action)
        result = StepResult()
        
        _true = False
        # Based on the simulator, next_state is true next state or imagined by the simulator
        if _true:
            print("taking true step")
            result.next_state, is_legal = self.take_real_state(state, action)
            print("next true state:", result.next_state.position)
        # Use true simulator for taking actions (only for eval)
        else:
            result.next_state, is_legal = self.make_next_position(state, action)
        
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state, always_obs=is_mdp)
        result.reward = self.make_reward(state, action, result.next_state, is_legal, always_obs=is_mdp)
        result.is_terminal = result.next_state.terminal
        return result, is_legal


    def reset_for_simulation(self):
        pass

    def sample_an_init_state(self):
        return BoxState(self.init_state)

    def create_root_historical_data(self, solver):
        return PositionData(self, self.init_state, solver)

    def belief_update(self, old_belief, action, observation):
        pass

    def get_max_undiscounted_return(self):
        return 1.0 # + 0.25 * 4
        # return 1

    def reset_for_epoch(self):
        self.t = 0

    def render(self):
        pass
