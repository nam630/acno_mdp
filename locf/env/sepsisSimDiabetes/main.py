import numpy as np
from sepsis_transition import SepsisEnv 
import pickle


def main():
    env = SepsisEnv(obs_cost=0., 
                    no_missingness=True,)
    T = {}
    R = np.zeros((720,))
    for action in range(8):
        import pdb;pdb.set_trace()
        T, R = env.step(action, T, R)
    pickle.dump(T, open("sepsis_T.obj", "wb"))
    pickle.dump(R, open("sepsis_R.obj", "wb"))
    print(T)
    print(R)

if __name__ == "__main__":
    main()
