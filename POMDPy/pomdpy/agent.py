from __future__ import print_function, division
import time
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider
# from experiments.scripts.pickle_wrapper import save_pkl
import numpy as np

module = "agent"

DIR = './exp/'
SAVE_K = '0'
if not os.path.exists(DIR + SAVE_K):
    os.makedirs(DIR + SAVE_K)

_eval = False

def action_dictionary():
    return {
            0: '0A_0E_0V',
            1: '0A_0E_1V',
            2: '0A_1E_0V',
            3: '0A_1E_1V',
            4: '1A_0E_0V',
            5: '1A_0E_1V',
            6: '1A_1E_0V',
            7: '1A_1E_1V',
            }

class Agent:
    """
    Train and store experimental results for different types of runs

    """

    def __init__(self, model, solver, is_mdp):
        """
        Initialize the POMDPY agent
        :param model:
        :param solver:
        :return:
        """
        if is_mdp == 1:
            self.solver_type = 'MDP'
        else:
            self.solver_type = 'POMDP'
        self.logger = logging.getLogger('POMDPy.Solver')
        self.model = model
        self.need_init = True
        self.cost = model.cost
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver

    def discounted_return(self, evaln=False):

        if self.model.solver == 'ValueIteration':
            solver = self.solver_factory(self)

            self.run_value_iteration(solver, 1000, evaln=evaln)

            if self.model.save:
                save_pkl(solver.gamma,
                         os.path.join(self.model.weight_dir,
                                      'VI_planning_horizon_{}.pkl'.format(self.model.planning_horizon)))
        
        elif not self.model.use_tf:
            self.multi_epoch()
        else:
            self.multi_epoch_tf()

        print('\n')
        console(2, module, 'epochs: ' + str(self.model.n_epochs))
        console(2, module, 'ave undiscounted return/step: ' + str(self.experiment_results.undiscounted_return.mean) +
                ' +- ' + str(self.experiment_results.undiscounted_return.std_err()))
        console(2, module, 'ave discounted return/step: ' + str(self.experiment_results.discounted_return.mean) +
                ' +- ' + str(self.experiment_results.discounted_return.std_err()))
        console(2, module, 'ave time/epoch: ' + str(self.experiment_results.time.mean))

        self.logger.info('env: ' + self.model.env + '\t' +
                         'epochs: ' + str(self.model.n_epochs) + '\t' +
                         'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) + ' +- ' +
                         str(self.experiment_results.undiscounted_return.std_err()) + '\t' +
                         'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                         ' +- ' + str(self.experiment_results.discounted_return.std_err()) +
                         '\t' + 'ave time/epoch: ' + str(self.experiment_results.time.mean))

    def multi_epoch_tf(self):
        import tensorflow as tf
        tf.set_random_seed(int(self.model.seed) + 1)

        with tf.Session() as sess:
            solver = self.solver_factory(self, sess)

            for epoch in range(self.model.n_epochs + 1):

                self.model.reset_for_epoch()

                if epoch % self.model.test == 0:
                    epoch_start = time.time()

                    print('evaluating agent at epoch {}...'.format(epoch))

                    # evaluate agent
                    reward = 0.
                    discounted_reward = 0.
                    discount = 0.7
                    belief = self.model.get_initial_belief_state()
                    step = 0
                    while step < self.model.max_steps:
                        action, v_b = solver.greedy_predict(belief)
                        step_result = self.model.generate_step(action, state=state)
                    
                        if not step_result.is_terminal:
                            belief = self.model.belief_update(belief, action, step_result.observation)

                        reward += step_result.reward
                        discounted_reward += discount * step_result.reward
                        discount *= self.model.discount

                        # show the step result
                        self.display_step_result(epoch, step_result)
                        step += 1
                        if step_result.is_terminal:
                            break

                    self.experiment_results.time.add(time.time() - epoch_start)
                    self.experiment_results.undiscounted_return.count += 1
                    self.experiment_results.undiscounted_return.add(reward)
                    self.experiment_results.discounted_return.count += 1
                    self.experiment_results.discounted_return.add(discounted_reward)

                    summary = sess.run([solver.experiment_summary], feed_dict={
                        solver.avg_undiscounted_return: self.experiment_results.undiscounted_return.mean,
                        solver.avg_undiscounted_return_std_dev: self.experiment_results.undiscounted_return.std_dev(),
                        solver.avg_discounted_return: self.experiment_results.discounted_return.mean,
                        solver.avg_discounted_return_std_dev: self.experiment_results.discounted_return.std_dev()
                    })
                    for summary_str in summary:
                        solver.summary_ops['writer'].add_summary(summary_str, epoch)

                    # TODO: save model at checkpoints
                else:
                    # train for 1 epoch
                    solver.train(epoch)

            if self.model.save:
                solver.save_alpha_vectors()
                print('saved alpha vectors!')

    def multi_epoch(self):
        eps = self.model.epsilon_start
        rewards = []
        temp = None
        self.model.reset_for_epoch()
        for i in range(self.model.n_epochs):
            # Reset the epoch stats
            start = time.time()
            self.results = Results()
            if self.model.solver == 'POMCP' or self.model.solver == 'MCP':
                eps, reward, temp = self.run_pomcp(i + 1, eps, temp)
                self.model.reset_for_epoch()
                rewards.append(reward)
            print('Runtime: ', time.time() - start)
            if self.experiment_results.time.running_total > self.model.timeout:
                print("TIMEOUT")
                break
            if i % 100 == 0:
                np.save(open(DIR+'{}/epoch_{}_rewards.npy'.format(SAVE_K, i), 'wb'), np.array(rewards))
        rewards = np.array(rewards)
        np.save(DIR+'pomcp_{}.npy'.format(SAVE_K), rewards)
        

    """
    Observe then plan uses the pre-learned transition and reward estimates (set observe_then_plan to True).
    ACNO-POMCP (observe while planning) updates the model estimates after every consecutive observed pair // otherwise both use same MC rollouts.
    """
    def run_pomcp(self, epoch, eps, temp=None, observe_then_plan=True):
        print('Epoch :', epoch)
        epoch_start = time.time()
        
        old_t = self.model.t_counts # 720 * 8 * 720
        old_r = self.model.r_counts # 720 * 8
        old_n = np.maximum(self.model.n_counts, np.ones(old_r.shape)) # 720 * 8
        
        if not observe_then_plan and (epoch % 100 == 0):
            np.save(open(DIR+'{}/epoch{}_T.npy'.format(SAVE_K, epoch), 'wb'), old_t)
            np.save(open(DIR+'{}/epoch{}_N.npy'.format(SAVE_K, epoch), 'wb'), self.model.n_counts)
            np.save(open(DIR+'{}/epoch{}_R.npy'.format(SAVE_K, epoch), 'wb'), old_r)
        
        # Create a new solver
        if epoch == 1:
            solver = self.solver_factory(self)
            temp = solver.fast_UCB.copy()
            self.need_init = False
        else:
            assert(temp is not None)
            solver = self.solver_factory(self)
            solver.fast_UCB = temp
        solver.model.t_estimates = old_t / old_n[:,:,np.newaxis]
        solver.model.r_estimates = old_r / old_n
        
        if observe_then_plan:
            terminal_states = np.load('terminal_states.npy')
            for terminal in terminal_states:
                solver.model.t_estimates[terminal, :, :] = 0.
                solver.model.t_estimates[terminal, :, terminal] = 1.
                solver.model.r_estimates[terminal, :] = 0.

            # POMDP MODEL (M')
            solver.model.r_estimates = np.load('rewards.npy')
            solver.model.t_estimates = np.load('transitions.npy')
        
        # Monte-Carlo start state
        state = solver.belief_tree_index.sample_particle()
        
        discounted_reward = 0
        reward = 0
        discount = 1.0 # starts with 1, drops by 0.7
        past_obs = state.position # always gives true state
        for i in range(self.model.max_steps):
            start_time = time.time()

            # Action will be of type Discrete Action
            # Need e-greedy selection
            action = solver.select_eps_greedy_action(eps, start_time, greedy_select=(not _eval))
            
            # Only run with _true if taking step in the true env (true performance calculated here)
            mdp = bool(self.model.solver == 'MCP')
            
            # state only used internally for simulation, action selection based on solver belief tree index (on obs)
            step_result, is_legal = self.model.generate_step(state, action, _true=True, is_mdp=mdp)
            start_time = time.time()
            discounted_reward += discount * (step_result.reward)
            reward += step_result.reward
            
            '''
            Observe while planning: if the last state was unobserved, then cannot learn a new transition model
            every time a new state is observed, use that state to upate the mle estimates
            '''
            if (not observe_then_plan) and (not _eval):
                if past_obs != 720 and action.bin_number < 8:
                    self.model.n_counts[past_obs, action.bin_number % 8] += 1
                    self.model.r_counts[past_obs, action.bin_number % 8] += step_result.reward
                    self.model.t_counts[past_obs, action.bin_number % 8, step_result.observation.position] += 1
            
            past_obs = step_result.observation.position
            start_time = time.time()
            discount *= self.model.discount # model discount = 0.7
            
            '''
            Set to true state from the env
            (pomcp solver uses obs for sampling particles, view L158 of pomdpy/solvers/belief_tree_solver.py 
            for how step_result is used in solver.update)
            '''
            state = step_result.next_state
            print("true state in epoch:", state.position)
            # Update epsilon every episode
            if eps > self.model.epsilon_minimum:
                eps *= self.model.epsilon_decay
            
            # Show the step result
            self.display_step_result(i, step_result)
            print(step_result.is_terminal)
            if not step_result.is_terminal or not is_legal:
                solver.update(step_result)
            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward, step_result.action, step_result.observation, step_result.next_state)

            if step_result.is_terminal or not is_legal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break
        
        self.results.time.add(time.time() - epoch_start)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        # print_divider('large')
        # solver.history.show()
        # self.results.show(epoch)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)
        print(discounted_reward)
        return eps, discounted_reward, temp

    
    '''
    Below code is not used for ACNO_MDP
    '''
    def run_value_iteration(self, solver, epoch, evaln=False):
        run_start_time = time.time()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        if evaln:
            # load the model
            import pickle
            solver.gamma = pickle.load(open('VI_planning_horizon.pkl', 'rb')) 
        else:
            solver.value_iteration(self.model.get_transition_matrix(),
                               self.model.get_observation_matrix(),
                               self.model.get_reward_matrix(),
                               self.model.planning_horizon)
            # save gamma's
            save_pkl(solver.gamma,
                os.path.join('VI_planning_horizon_{}_adjusted.pkl'.format(self.model.planning_horizon)))

        '''
        Planning is done fully offline
        Only need to take steps in the env
        '''
        rewards = []
        for itr in range(epoch):
            reward = 0
            b = self.model.get_initial_belief_state()
            state = self.model.get_start_state()
            for i in range(self.model.max_steps):
                # TODO: record average V(b) per epoch
                action, v_b = solver.select_action(b, solver.gamma)
                step_result = self.model.generate_step(action, state=state) 
                state = step_result.next_state.copy()
                if not step_result.is_terminal:
                    b = self.model.belief_update(b, action, step_result.observation)
                reward += step_result.reward
                discounted_reward += discount * step_result.reward
                discount *= self.model.discount

                # show the step result
                # self.display_step_result(i, step_result)
                if step_result.is_terminal:
                    rewards.append(reward)
                    # console(3, module, 'Terminated after episode step ' + str(i + 1))
                    break

            self.results.time.add(time.time() - run_start_time)
            self.results.update_reward_results(reward, discounted_reward)
            # Pretty Print results
            # self.results.show(epoch)
            # console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
            # print_divider('medium')

            self.experiment_results.time.add(self.results.time.running_total)
            self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
            self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
            self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
            self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)
        rewards = np.array(rewards)
        print('AVG REWARD:', np.mean(rewards))
        print('Sucess num: ', sum(rewards > 0))

    @staticmethod
    def display_step_result(step_num, step_result):
        """
        Pretty prints step result information
        :param step_num:
        :param step_result:
        :return:
        """
        console(3, module, 'Step Number = ' + str(step_num))
        console(3, module, 'Step Result.Action = ' + step_result.action.to_string())
        console(3, module, 'Step Result.Observation = ' + step_result.observation.to_string())
        # console(3, module, 'Step Result.Next_State = ' + step_result.next_state.to_string())
        console(3, module, 'Step Result.Reward = ' + str(step_result.reward))


class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, epoch):
        print_divider('large')
        print('\tEpoch #' + str(epoch) + ' RESULTS')
        print_divider('large')
        console(2, module, 'discounted return statistics')
        print_divider('medium')
        self.discounted_return.show()
        print_divider('medium')
        console(2, module, 'undiscounted return statistics')
        print_divider('medium')
        self.undiscounted_return.show()
        print_divider('medium')
        console(2, module, 'Time')
        print_divider('medium')
        self.time.show()
        print_divider('medium')
