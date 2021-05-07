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
C =  20 # might need to make larger? 
H = 5
Bv = np.sqrt(2 * C)
Bp = H * Bv
J = H * C / 3.
JB = 4 * J + Bp
A = 8
S = 720
INIT_STATE = 256 # sepsis patient state
N_euler = 5000 # number of euler episodes

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

# n(s,a) = max(1, n[(s,a)])

'''
Confidence interval for transition * value estimate (check which s to use)
'''
def find_phi(p_hat, v, n_sa):
    mu = p_hat @ v
    var_p_hat = p_hat @ ((v - mu)**2)
    return np.sqrt(2 * C * var_p_hat / n_sa) + H * C / (3 * max(1, n_sa - 1))

'''
Lines 6~9 for a particular (s,a)
'''
def euler_q(pi, s, a, p_sum, n, r_sum, v_upper, v_lower, t, b_r):
    n_s_a = max(1, n[(s,a)])
    p_hat = p_sum[(s, a)] / n_s_a
    v_dist = np.sqrt(sum([p_hat[i] * (v_upper[i] - v_lower[i])**2 for i in range(len(p_hat))]))
    b_pv = 1./np.sqrt(n_s_a) * (JB / np.sqrt(n_s_a) + Bv * v_dist) + find_phi(p_hat, v_upper, n_s_a)
    pi_act = pi[(s, t)]
    r_hat = r_sum[(s,pi_act)] / max(1, n[(s,pi_act)])
    return min(H-t, r_hat + b_r + p_hat @ v_upper + b_pv)

'''
Return new_v_upper(s) and new_v_lower(s) for a chosen s
Lines 11~14
'''
def euler_s(s, pi, n, p_sum, r_sum, r_var, v_upper, v_lower, t):
    q = np.zeros((A,))
    for a in range(A):
        # b_k^r(s,a)
        r_var_hat = 0.
        if len(r_var[(s,a)]) > 0:
            r_var_hat = np.std(r_var[(s, a)]) ** 2
        
        n_s_a = max(1, n[(s,a)])
        b_r = np.sqrt(2 * C * r_var_hat / n_s_a) + 7 * C / (3 * max(1, n[(s,a)] - 1))
        q[a] = euler_q(pi, s, a, p_sum, n, r_sum, v_upper, v_lower, t, b_r)
    
    pi_act = np.random.choice(np.argwhere(q == np.amax(q)).flatten()) # randomly choose if have same q vals
    pi[(s, t)] = pi_act
    new_v_upper = q[pi_act]
    
    n_s_a = max(1, n[(s, pi_act)])
    p_hat = p_sum[(s, pi_act)] / n_s_a
    v_dist = np.sqrt(sum([p_hat[i] * (v_upper[i] - v_lower[i])**2 for i in range(len(p_hat))]))
    b_pv = 1./np.sqrt(n_s_a) * (JB / np.sqrt(n_s_a) + Bv * v_dist) + find_phi(p_hat, v_lower, n_s_a)
    r_hat = r_sum[(s, pi_act)] / n_s_a
    r_var_hat = 0.
    if len(r_var[(s, pi_act)]) > 0:
        r_var_hat = np.std(r_var[(s, pi_act)]) ** 2
    b_r_argmax = np.sqrt(2 * C * r_var_hat / n_s_a) + 7 * C / (3 * max(1, n[(s, pi_act)] - 1))
    new_v_lower = max(0, r_hat - b_r_argmax + p_hat @ v_lower - b_pv)
    
    return pi, new_v_upper, new_v_lower

'''
Wrapper for Lines 4~16
'''
def euler_iter(pi, n, p_sum, r_sum, r_var):
    v_upper = np.zeros((S,))
    v_lower = np.zeros((S,))
    for t in range(H, 0, -1): # t from H~1
        start = time.time()
        new_v_upper = v_upper.copy()
        new_v_lower = v_lower.copy()
        for state in range(S):
            pi, _upper, _lower = euler_s(state, pi, n, p_sum, r_sum, r_var, v_upper, v_lower, t)
            new_v_upper[state] = _upper
            new_v_lower[state] = _lower
        v_upper = new_v_upper
        v_lower = new_v_lower
        print("one step: ", time.time() - start)
    return pi

def initialize():
    pi = {} # randomly assign action?
    n = {}
    p_sum = {}
    r_sum = {}
    r_var = {}
    for state in range(S):
        for action in range(A):
            n[(state, action)] = 0.
            p_sum[(state, action)] = np.zeros((S,))
            r_sum[(state, action)] = 0.
            r_var[(state, action)] = []
    for state in range(S):
        for t in range(1, H+1):
            pi[(state, t)] = np.random.choice(A, 1)[0] # randomly assign 
    return pi, n, p_sum, r_sum, r_var

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

def execute_pi(pi, n, p_sum, r_sum, r_var):
    # initialize with a fixed start state 
    t = H
    env = SepsisEnv(obs_cost=0, no_missingness=True)
    state = env.reset(INIT_STATE)
    while t > 0:
        act = pi[(state, t)]
        next_state, reward, done = transition(env, act)
        r_sum[(state, act)] += reward
        r_var[(state, act)].append(reward)
        n[(state, act)] += 1
        p_sum[(state, act)][next_state] += 1
        state = next_state
        t -= 1
        if done:
            return reward, n, p_sum, r_sum, r_var
    # return n, p_sum, r_sum, r_var

'''
Main function 
initializes to all 0's 
also takes actions in the env and updates n, p_sum, r_sum
'''
def euler(verbose=True):
    stats = []
    # keep r_var separately from r_sum
    pi, n, p_sum, r_sum, r_var = initialize()
    for k in range(N_euler):
        start = time.time()
        pi  = euler_iter(pi, n, p_sum, r_sum, r_var)
        end = time.time()
        # print('one eps: ', end - start)
        # take actions using pi
        reward, n, p_sum, r_sum, r_var = execute_pi(pi, n, p_sum, r_sum, r_var)
        stats.append(reward)
        print("one eps: ", end - start, "reward: ", reward)
    if verbose:
        print(stats)
        pickle.dump(stats, open("reward_1.obj","wb"))
        pickle.dump(pi, open("policy_1.obj", "wb"))


def explore_pi(pi, n, p_sum, r_sum, r_var, n_global, p_global, r_global, goal_s, goal_a):
    # initialize with a fixed start state 
    dummy = False # test euler on a dummy env
    t = H
    if dummy:
        state = 0 # always start from a fixed 0 state
    else:
        env = SepsisEnv(obs_cost=0, no_missingness=True)
        state = env.reset(INIT_STATE)

    while t > 0:
        act = pi[(state, t)]
        if not dummy:
            next_state, _, done = transition(env, act)
        else:
            next_state = np.random.choice(S, p=T[act][state, :])
            done = False
        reward = 0.
        if next_state == goal_s:
            reward = 1.0
            done = True
        r_sum[(state, act)] += reward
        r_var[(state, act)].append(reward)
        n[(state, act)] += 1
        p_sum[(state, act)][next_state] += 1
        state = next_state
        if done:
            break
        t -= 1
    if state == goal_s and t > 0:
        if not dummy:
            next_state, reward, done = transition(env, goal_a)
        else:
            next_state = np.random.choice(S, p=T[goal_a][goal_s, :])
            reward = R[goal_s, goal_a]
        n_global[(goal_s, goal_a)] += 1
        p_global[(goal_s, goal_a)][next_state] += 1
        r_global[(goal_s, goal_a)] += reward
    return reward, n, p_sum, r_sum, r_var, n_global, p_global, r_global 

def euler_explore(n_global, p_global, r_global, goal_s, goal_a):
    # keep r_var separately from r_sum
    pi, n, p_sum, r_sum, r_var = initialize()
    for k in range(N_euler):
        start = time.time()
        pi  = euler_iter(pi, n, p_sum, r_sum, r_var)
        end = time.time()
        reward, n, p_sum, r_sum, r_var, n_global, p_global, r_global = explore_pi(pi, n, p_sum, r_sum, r_var, n_global, p_global, r_global, goal_s, goal_a)
    return n_global, p_global, r_global

def pomdp(goal, verbose=True):
    _, n_global, p_global, r_global, _ = initialize()
    # for s in range(S):
    for a in range(A):
        n_global, p_global, r_global = euler_explore(n_global, p_global, r_global, goal, a)
    if verbose:
        parent = "pomdp/{}".format(goal)
        if not os.path.exists(parent):
            os.makedirs(parent)
        pickle.dump(n_global, open(parent+"/n_pomdp.obj", "wb"))
        pickle.dump(p_global, open(parent+"/p_pomdp.obj", "wb"))
        pickle.dump(r_global, open(parent+"/r_pomdp.obj", "wb"))


def main(goal):
    # parent = "pomdp_{}".format(goal)
    # pickle.dump(T, open("pomdp_{}/true_t.obj", "wb"))
    # pickle.dump(R, open("pomdp_{}/true_r.obj", "wb"))
    # pomdp(goal)
    euler()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', default=0, type=int)
    args = parser.parse_args()
    main(args.goal)
