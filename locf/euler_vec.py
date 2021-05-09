import numpy as np
import sys
sys.path.append('/next/u/hjnam/locf/env/sepsisSimDiabetes')
from env.sepsisSimDiabetes.sepsis_tabular import SepsisEnv 
import time
import pickle
import os

'''
Constant C for delta, Bp, Bv, J, ln(4SAT/delta)
'''
C =  3 # might need to make larger? 
H = 5
Bv = np.sqrt(2 * C)
Bp = H * Bv
J = H * C / 3.
JB = 4 * J + Bp
A = 8
S = 720
INIT_STATE = 256 # sepsis patient state
N_euler = 100 # 5000 # number of euler episodes
LOG_K = 50
EVAL_N = 100
verbose = False
DIR = 'pomdp_0508/100/'

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
    v_dist = np.sqrt(p_star @ (v_upper - v_lower)**2)
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
    state, reward, done, info = env.step(a) 
    if done:
        if reward == 0:
            reward = 0.5 
        elif reward < 0:
            reward = 0
        elif reward == 1:
            reward = 1
    return state, reward, done
    # return np.random.choice(S, 1)[0], np.random.normal(0, 1, 1)[0]

def execute_pi(pi, n_mat, p_sum, r_sum, r_var):
    # initialize with a fixed start state 
    t = H
    env = SepsisEnv(obs_cost=0, no_missingness=True)
    state = env.reset(INIT_STATE)
    while t > 0:
        # act = pi[(state, t)]
        act = int(pi[state, t-1])
        next_state, reward, done = transition(env, act)
        # r_sum[(state, act)] += reward
        r_sum[state, act] += reward
        r_var[(state, act)].append(reward)
        n_mat[state, act] += 1
        p_sum[state, act, next_state] += 1
        # n[(state, act)] += 1
        # p_sum[(state, act)][next_state] += 1
        state = next_state
        t -= 1
        if done:
            return reward, n_mat, p_sum, r_sum, r_var
    # return reward, n, p_sum, r_sum, r_var

def find_eval_pi(n_mat, p_sum, r_sum):
    pi = np.zeros((S, H)) # randomly assign action?
    n_mat = np.maximum(np.ones((S, A)), n_mat)
    p_hat = p_sum / n_mat[:,:,np.newaxis]
    r_hat = r_sum/ n_mat
    v = np.zeros((S))
    for t in range(H, 0, -1): # t from H~1
        Q = r_hat + p_hat @ v
        pi_t = np.argmax(Q, 1) 
        v = np.max(Q, 1)
        pi[:, t-1] = pi_t 
    return pi


def evaluate_pi(pi):
    # initialize with a fixed start state 
    t = H
    env = SepsisEnv(obs_cost=0, no_missingness=True)
    state = env.reset(INIT_STATE)
    reward = 0.
    while t > 0:
        # act = pi[(state, t)]
        act = int(pi[state, t-1])
        next_state, rew, done = transition(env, act)
        reward += rew
        state = next_state
        t -= 1
        if done:
            return reward
    return reward

'''
Main function 
initializes to all 0's 
also takes actions in the env and updates n, p_sum, r_sum
'''
def euler():
    stats = []
    stats_eval = []
    # keep r_var separately from r_sum
    n, p_sum, r_sum, r_var = initialize()
    for k in range(N_euler):
        start = time.time()
        pi = euler_iter(n, p_sum, r_sum, r_var)
        end = time.time()
        # take actions using pi
        reward, n, p_sum, r_sum, r_var = execute_pi(pi, n, p_sum, r_sum, r_var)
        # reward, n, p_sum, r_sum, r_var = execute_pi(pi, n, p_sum, r_sum, r_var)
        stats.append(reward)
        if verbose:
            print("one eps: ", end - start, "reward: ", reward)
            if (k % LOG_K) == 0:
                eval_pi = find_eval_pi(n, p_sum, r_sum)  
                eval_rewards = [evaluate_pi(eval_pi) for j in range(EVAL_N)]
                stats_eval.append(np.mean(eval_rewards))
                print("one eps: ", end - start, "reward: ", stats_eval[-1])
    if verbose:
        # print(stats_eval)
        # print(stats)
        pickle.dump(stats, open(DIR + "obsvd_reward_vec.obj","wb"))
        pickle.dump(stats_eval, open(DIR + "eval_reward_vec.obj","wb"))
        pickle.dump(pi, open(DIR + "policy_vec.obj", "wb"))
        pickle.dump(r_sum, open(DIR + "reward_sum.obj", "wb"))
        pickle.dump(p_sum, open(DIR + "prob_sum.obj", "wb"))
        pickle.dump(n, open(DIR + "n_visits.obj", "wb"))

def explore_pi(pi, n, p_sum, r_sum, r_var, n_global, p_global, r_global, goal_s, goal_a):
    env = SepsisEnv(obs_cost=0, no_missingness=True)
    state = env.reset(INIT_STATE)
    t = H
    while t > 0:
        act = int(pi[state, t-1])
        next_state, _, done = transition(env, act)
        reward = 0.
        if next_state == goal_s:
            reward = 1.0
            done = True
        r_sum[state, act] += reward
        r_var[state, act].append(reward)
        n[state, act] += 1
        p_sum[state, act, next_state] += 1
        state = next_state
        if done:
            break
        t -= 1
    if state == goal_s:
        next_state, reward, done = transition(env, goal_a)
        n_global[goal_s, goal_a] += 1
        p_global[goal_s, goal_a, next_state] += 1
        r_global[goal_s, goal_a] += reward
    return n, p_sum, r_sum, r_var, n_global, p_global, r_global 

def euler_explore(n_global, p_global, r_global, goal_s, goal_a):
    # keep r_var separately from r_sum
    n, p_sum, r_sum, r_var = initialize()
    for k in range(N_euler):
        start = time.time()
        pi = euler_iter(n, p_sum, r_sum, r_var)
        end = time.time()
        n, p_sum, r_sum, r_var, n_global, p_global, r_global = explore_pi(pi, n, p_sum, r_sum, r_var, \
                                                                        n_global, p_global, r_global, goal_s, goal_a)
    return n_global, p_global, r_global

def pomdp(start, end, save=True):
    n_global, p_global, r_global, _ = initialize()
    for s in range(start, end):
        for a in range(A):
            start_t = time.time()
            n_global, p_global, r_global = euler_explore(n_global, p_global, r_global, s, a)
            print('Finished state ', s, ' action ', a, ' in ', time.time() - start_t)
    if save:
        parent = DIR + "s_{}_e_{}".format(start, end)
        pickle.dump(n_global, open(parent+"_n.obj", "wb"))
        pickle.dump(p_global, open(parent+"_p.obj", "wb"))
        pickle.dump(r_global, open(parent+"_r.obj", "wb"))

def main(args):
    # parent = "pomdp_{}".format(goal)
    # pickle.dump(T, open("pomdp_{}/true_t.obj", "wb"))
    # pickle.dump(R, open("pomdp_{}/true_r.obj", "wb"))
    if args.explore:
        pomdp(args.start, args.end)
    else:
        euler()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=2, type=int)
    parser.add_argument('--explore', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
