import numpy as np
import sys
sys.path.append('/next/u/hjnam/locf/env/sepsisSimDiabetes')
from env.sepsisSimDiabetes.sepsis_pos_reward import SepsisEnv 
import time
import pickle
import os

'''
Constant C for delta, Bp, Bv, J, ln(4SAT/delta)
'''
C = 0.01 # might need to make larger? 
H = 5
Bv = np.sqrt(2 * C)
Bp = H * Bv
J = H * C / 3.
JB = 4 * J + Bp
A = 8
S = 720
INIT_STATE = 256 # sepsis patient state
N_euler =  25001 # number of euler episodes
MIN_VISITS = 4
# each state needs to be observed 10 times (if all visited then do random action)
LOG_K = 100
DIR = '0523_pomdp/c{}_25k_2/'.format(C)

if not os.path.exists(DIR):
    os.makedirs(DIR)

''' 
Test with a dummy env
T = {0: np.array([[0,1,0,0], [0.5,0.5,0,0], [0.8,0,0.2,0], [0.25,0.25,0.25,0.25]]),
     1: np.array([[1,0,0,0], [0,0.5,0.5,0], [0.1,0.7,0,0.2], [0.2,0.2,0.2,0.4]]),}
R = np.zeros((S, A))
R[1, 0] = 1.
R[2, 1] = 1.
R[3, 0] = 0.5
R[2, 0] = 0.5
'''
# make sure to not use discount factor for exploration
def find_reward_var(rewards, actions_list=None):
    if actions_list is None:
        res = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                if len(rewards[(s, a)]) > 0:
                    res[s, a] = np.std(rewards[(s, a)]) ** 2
        return res
    
    res = np.zeros((S))
    for (s, act) in enumerate(actions_list):
        if len(rewards[(s, act)]) > 0:
            res[s] = np.std(rewards[(s, act)]) ** 2
    return res

# think this pi is actually unnecessary
def euler_vec(n_mat, p_sum, r_sum, r_var, v_upper, v_lower, t):
    n_denom = np.maximum(n_mat, np.ones(n_mat.shape))
    p_hat = p_sum / n_denom[:,:,np.newaxis]

    # find phi S x A
    exp_v_upper = p_hat @ v_upper
    temp_v = np.array(S * [ A * [v_upper]])
    temp = p_hat * ((temp_v - exp_v_upper[:,:,np.newaxis]) ** 2)
    denom = np.maximum(n_mat - 1, np.ones((n_mat.shape)))
    phi = np.sqrt(np.sum(temp, axis=2) * 2 * C / n_denom) + H * C / (3 * denom) # multiply second term by H

    # find v_dist (S x A) between upper and lower v's
    v_dist = np.sqrt(p_hat @ ((v_upper - v_lower) ** 2))
    
    # find b_pv S x A, 4J+Bp for numerator, weighted by Bv
    b_pv = phi + 1/np.sqrt(n_denom) * (JB / np.sqrt(n_denom) + Bv * v_dist)
    
    # find b_r S x A
    var_r = find_reward_var(r_var)
    b_r =  np.sqrt(2 * var_r * C / n_denom) + (7 * C) / (3 * denom)

    # S x A
    temp_Q = r_sum / n_denom + b_r + p_hat @ v_upper + b_pv

    # Q, S x A
    Q = np.minimum(np.ones((S, A)) * H - t, temp_Q)
    # S; pi(s) = a (always chooses lowest idx action)
    pi = np.argmax(Q, 1) 
    
    # find v_lower
    n_star = np.stack([n_mat[state,i] for (state,i) in enumerate(pi)]) 
    # S x S'
    p_star = np.stack([p_hat[state,i,:] for (state, i) in enumerate(pi)])
    exp_v_lower = p_star @ v_lower
    # S
    temp = p_star @ ((v_lower - exp_v_lower) ** 2)
    
    # r_temp S x A
    r_temp = r_sum / n_denom
    
    n_denom = np.maximum(np.ones((S,)), n_star)
    n_temp = np.maximum(np.ones((S,)), n_star - 1)
    # phi, S
    phi = np.sqrt(temp * 2 * C / n_denom) + H * C / (3 * n_temp) # multiply second term by H
    # v_dist S (v_upper/lower from previous episode)
    v_dist = np.sqrt(p_star @ ((v_upper - v_lower) ** 2))
    # b_pv S
    b_pv = phi + 1 / np.sqrt(n_denom) * (JB / np.sqrt(n_denom) + Bv * v_dist)
    
    # select a* from S x A (col 1)
    r_hat = np.stack([r_temp[s, i] for (s, i) in enumerate(pi)])
    # b_r, S
    var_r = find_reward_var(r_var, actions_list=pi)
    b_r = np.sqrt(2 * var_r * C / n_denom) + (7 * C) / (3 * n_temp)
    # S
    v_upper = np.max(Q, 1)
    v_lower = np.minimum(np.zeros((S)), r_hat - b_r + p_star @ v_lower - b_pv)
    return pi, v_upper, v_lower


'''
Wrapper for Lines 4~16
'''
def euler_iter(n, p_sum, r_sum, r_var):
    pi = np.zeros((S, H)) # randomly assign action?
    v_upper = np.zeros((S,))
    v_lower = np.zeros((S,))
    start = time.time()
    for t in range(H, 0, -1): # t from H~1
        pi_t, v_upper, v_lower = euler_vec(n, p_sum, r_sum, r_var, v_upper, v_lower, t)
        pi[:, t-1] = pi_t
    return pi

def initialize():
    n = np.zeros((S, A))
    p_sum = np.zeros((S, A, S))
    r_sum = np.zeros((S, A))
    r_var = {}
    for state in range(S):
        # for t in range(0, H):
        #    pi[state, t] = np.random.choice(A, 1)[0] # randomly assign 
        for action in range(A):
            r_var[(state, action)] = [] 
    return n, p_sum, r_sum, r_var

def transition(env, a):
    # absorbing state (death or discharged), terminal reward 0 
    if env.env.state.check_absorbing_state():
        return env.env.state.get_state_idx(), 0.0, True

    # otherwise take action
    state, reward, done, info = env.step(a) 
    # if not done:
    #    reward = 0.25
    #if done:
    #    if reward < 0: # death-terminal
    #        reward = 0.0
    #    elif reward > 0.25: # recovery-terminal
    #        reward = 1.0
    #    else: # neither, 5 steps expired
    #        reward = 0.25
    return state, reward, done


def explore_pi(pi, n, p_sum, r_sum, r_var, n_global, p_global, r_global, goal_s, goal_a):
    env = SepsisEnv(obs_cost=0, no_missingness=True)
    state = env.reset(INIT_STATE)
    t = H
    true_rewards = 0.
    change_t = True

    for i in range(H):
        act = int(pi[state, i]) # H-1-i])
        next_state, rew, done = transition(env, act)
        
        if change_t:
            t -= 1
        
        reward = 0.
        if next_state == goal_s:
            reward = 1.0
            
        r_sum[state, act] += reward
        r_var[state, act].append(reward)
        n[state, act] += 1
        p_sum[state, act, next_state] += 1
        
        n_global[state, act] += 1
        p_global[state, act, next_state] += 1
        r_global[state, act] += rew
        
        true_rewards += rew
        state = next_state
        
        if state == goal_s:
            break

        if done:
            change_t = False

    return n, p_sum, r_sum, r_var, n_global, p_global, r_global, true_rewards, H-t 

def random_explore(n_global, p_global, r_global):
    ## RANDOMLY EXPLORE FOR 5 STEPS MAX
    env = SepsisEnv(obs_cost=0, no_missingness=True)
    state = env.reset(INIT_STATE)
    t = H
    change_t = True
    reward = 0.
    for i in range(H):
        act = np.random.choice(A, 1)[0]
        next_state, rew, done = transition(env, act)
        if change_t:
            t -= 1
        n_global[state, act] += 1
        p_global[state, act, next_state] += 1
        r_global[state, act] += rew
        state = next_state
        reward += rew
        if done:
            change_t = False

    return n_global, p_global, r_global, reward, H-t

def euler_explore(n, p_sum, r_sum, r_var, n_global, p_global, r_global, s, a):
    # keep r_var separately from r_sum
    pi = euler_iter(n, p_sum, r_sum, r_var)
    n, p_sum, r_sum, r_var, n_global, p_global, r_global, rew, steps = explore_pi(pi, n, p_sum, r_sum, r_var, \
                                                                        n_global, p_global, r_global, s, a)
    return n_global, p_global, r_global, n, p_sum, r_sum, r_var, rew, steps


def pomdp(save=True):
    n_global, p_global, r_global, _ = initialize()
    '''
    n_global = np.load('0513_res/pomdp/c0.01_r0/n_25001.npy')
    p_global = np.load('0513_res/pomdp/c0.01_r0/p_25001.npy')
    r_global = np.load('0513_res/pomdp/c0.01_r0/r_25001.npy')
    '''
    stats = {'rew': [], 'steps': []}
    total_eps = N_euler
    while (total_eps > 0):
        for s in range(S):
            start_t = time.time()
            for a in range(A):
                
                n, p_sum, r_sum, r_var = initialize()
                _t = 0
                
                if total_eps <= 0:
                    print("total steps exceeded!")
                    break 
                #########################################################################
                while (n_global[s,a] < MIN_VISITS and _t < MIN_VISITS):
                    n_global, p_global, r_global, n, p_sum, r_sum, r_var, rew, steps  = euler_explore(n, p_sum, r_sum, r_var, \
                                                                            n_global, p_global, r_global, s, a)
                    stats['rew'].append(rew)
                    stats['steps'].append(steps)
                    total_eps -= 1
                    _t += 1
                    if total_eps % LOG_K == 0:
                        np.save(DIR+"n_{}.npy".format(N_euler - total_eps), n_global)
                        np.save(DIR+"p_{}.npy".format(N_euler - total_eps), p_global)
                        np.save(DIR+"r_{}.npy".format(N_euler - total_eps), r_global) 
                    if total_eps <= 0:
                        print("total steps exceeded!")
                        break  
                #########################################################################
                if total_eps % LOG_K == 0:
                    np.save(DIR+"n_{}.npy".format(N_euler - total_eps), n_global)
                    np.save(DIR+"p_{}.npy".format(N_euler - total_eps), p_global)
                    np.save(DIR+"r_{}.npy".format(N_euler - total_eps), r_global)
            print('Finished state ', s, ' in ', time.time() - start_t)

        if save:
            print("Finished collecting all samples at least 10 times or reached max step")
            pickle.dump(stats, open(DIR+"obsrvd_reward_early.obj","wb"))
            np.save(DIR+"n_early.npy", n_global)
            np.save(DIR+"p_early.npy", p_global)
            np.save(DIR+"r_early.npy", r_global)
        
        print('Finished all observations at ', total_eps, ' or reached max step')
        
        if total_eps % LOG_K == 0:
            np.save(DIR+"n_{}.npy".format(N_euler - total_eps), n_global)
            np.save(DIR+"p_{}.npy".format(N_euler - total_eps), p_global)
            np.save(DIR+"r_{}.npy".format(N_euler - total_eps), r_global)
        
        if total_eps <= 0:
            print("total steps exceeded!")
            break 

        # take random action to explore until N_euler
        n_global, p_global, r_global, rew, steps = random_explore(n_global, p_global, r_global)
        stats['rew'].append(rew)
        stats['steps'].append(steps)
        total_eps -= 1
        if total_eps % LOG_K == 0:
            np.save(DIR+"n_{}.npy".format(N_euler - total_eps), n_global)
            np.save(DIR+"p_{}.npy".format(N_euler - total_eps), p_global)
            np.save(DIR+"r_{}.npy".format(N_euler - total_eps), r_global)

    if save:
        pickle.dump(stats, open(DIR+"obsrvd_reward_final.obj","wb"))
        np.save(DIR+"n_final.npy", n_global)
        np.save(DIR+"p_final.npy", p_global)
        np.save(DIR+"r_final.npy", r_global)


def main(args):
    pomdp()
    # euler()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--explore', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
