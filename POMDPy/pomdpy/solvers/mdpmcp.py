from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
import time
import numpy as np
from pomdpy.util import console
from pomdpy.action_selection import ucb_action
from .belief_tree_solver import BeliefTreeSolver
import h5py

module = "mcp"


class MCP(BeliefTreeSolver):
    """
    Monte-Carlo Tree Search implementation, from MCP
    """

    # Dimensions for the fast-UCB table
    UCB_N = 10000
    UCB_n = 100

    def __init__(self, agent):
        """
        Initialize an instance of the POMCP solver
        :param agent:
        :param model:
        :return:
        """
        super(MCP, self).__init__(agent)
        '''
        import os
        self.file = 'res/0411/mdp_c_20/' # run with cost -1 per observation (empirical sim)
        if not os.path.exists(self.file):
            os.makedirs(self.file)
        self.file += 'patient_256.h5'
        '''
        # Pre-calculate UCB values for a speed-up
        self.fast_UCB = [[None for _ in range(MCP.UCB_n)] for _ in range(MCP.UCB_N)]

        for N in range(MCP.UCB_N):
            for n in range(MCP.UCB_n):
                if n is 0:
                    self.fast_UCB[N][n] = np.inf
                else:
                    self.fast_UCB[N][n] = agent.model.ucb_coefficient * np.sqrt(old_div(np.log(N + 1), n))

    @staticmethod
    def reset(agent):
        """
        Generate a new POMCP solver

        :param agent:
        Implementation of abstract method
        """
        return MCP(agent)

    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        """
        Look up and return the value in the UCB table corresponding to the params
        :param total_visit_count:
        :param action_map_entry_visit_count:
        :param log_n:
        :return:
        """
        assert self.fast_UCB is not None
        if total_visit_count < MCP.UCB_N and action_map_entry_visit_count < MCP.UCB_n:
            return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

        if action_map_entry_visit_count == 0:
            return np.inf
        else:
            return self.model.ucb_coefficient * np.sqrt(old_div(log_n, action_map_entry_visit_count))

    def select_eps_greedy_action(self, eps, start_time):
        """
        Starts off the Monte-Carlo Tree Search and returns the selected action. If the belief tree
                data structure is disabled, random rollout is used.
        """
        if self.disable_tree:
            self.rollout_search(self.belief_tree_index)
        else:
            self.monte_carlo_approx(eps, start_time)
        
        actions = list(self.belief_tree_index.action_map.entries.values())
        act_vals = np.zeros((len(actions),))
        
        for i in range(len(actions)):
            idx = actions[i].get_action().bin_number
            act_vals[int(idx)] = actions[i].mean_q_value
       
        particles = self.belief_tree_index.state_particles
        p = []
        for i in range(len(particles)):
            p.append(particles[i].position)
        p = np.array(p)
        # print('S:', self.belief_tree_index.state_particles[0].position, 'e: ', eps, act_vals)
        # print("=================")
        p = np.expand_dims(p, axis=0)
        act_vals = np.expand_dims(act_vals, axis=0)
        '''
        first = bool(eps>0.85)
        if first:
            with h5py.File(self.file,'w') as f:
                f.create_dataset('state', data=p, maxshape=(None, p.shape[1]))
                f.create_dataset('action_val', data=act_vals, maxshape=(None, act_vals.shape[1]))
        else:
            with h5py.File(self.file, 'a') as f:
                f['state'].resize((f['state'].shape[0] + 1), axis=0)
                f['state'][-1:] = p
                f['action_val'].resize((f['action_val'].shape[0] + 1), axis=0)
                f['action_val'][-1:] = act_vals
        '''
        return ucb_action(self, self.belief_tree_index, True)

    def simulate(self, belief_node, eps, start_time):
        """
        :param belief_node:
        :return:
        """
        return self.traverse(belief_node, 0, start_time)

    def traverse(self, belief_node, tree_depth, start_time):
        delayed_reward = 0

        state = belief_node.sample_particle()

        # Time expired
        if time.time() - start_time > self.model.action_selection_timeout:
            console(4, module, "action selection timeout")
            return 0

        action = ucb_action(self, belief_node, False)

        # Search horizon reached
        if tree_depth >= self.model.max_depth:
            console(4, module, "Search horizon reached")
            return 0

        step_result, is_legal = self.model.generate_step(state, action, is_mdp=True)

        child_belief_node = belief_node.child(action, step_result.observation)
        if child_belief_node is None and not step_result.is_terminal and belief_node.action_map.total_visit_count > 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)

        if not step_result.is_terminal or not is_legal:
            tree_depth += 1
            if child_belief_node is not None:
                # Add S' to the new belief node
                # Add a state particle with the new state
                if child_belief_node.state_particles.__len__() < self.model.max_particle_count:
                    child_belief_node.state_particles.append(step_result.next_state)
                delayed_reward = self.traverse(child_belief_node, tree_depth, start_time)
            else:
                delayed_reward = self.rollout(belief_node)
            tree_depth -= 1
        else:
            console(4, module, "Reached terminal state.")

        # delayed_reward is "Q maximal"
        # current_q_value is the Q value of the current belief-action pair
        action_mapping_entry = belief_node.action_map.get_entry(action.bin_number)

        q_value = action_mapping_entry.mean_q_value

        # off-policy Q learning update rule
        q_value += (step_result.reward + (self.model.discount * delayed_reward) - q_value)

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        # Add RAVE ?
        return q_value
