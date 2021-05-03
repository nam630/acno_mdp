# acno_mdp

Credits:
- DVRL encoder & a2c is slighlty adapted from https://github.com/maximilianigl/DVRL (Igl et al)
- Monte-Carlo Tree Search and POMDP tree search code is from https://github.com/pemami4911/POMDPy.
@ARTICLE{emami2015pomdpy,
  author = {Emami, Patrick and Hamlet, Alan J., and Crane, Carl},
  title = {POMDPy: An Extensible Framework for Implementing POMDPs in Python},
  year = {2015},
}
- Sepsis simulator code is from https://github.com/clinicalml/gumbel-max-scm.

Conda packages and versions used for generating the reported results are shared in conda.yml (Note not all the packages are needed).

To run the known observation belief encoder
1. cartpole (can adjust observation cost in cofig_file)
python known_obs/code/main_known.py -p with environment.config_file=cartpole_ver3.yaml
2. mountain hike (can adjust observation cost in config file)
python known_obs/code/main_known.py -p with environment.config_file=mountainHike_ver2.yaml

To run the default DVRL belief encoder (also need to manually set env_id in code/envs.py 'make_env' -- review inline comments)
1. cartpole (need to set obs_cost in custom_cartpole/envs/AdvancedCartPole.py obs_cost)
python ./code/main.py -p with environment.config_file=cartpole_ver2.yaml algorithm.use_particle_filter=True log.filename='temp/'
2. mountain (need to set obs_cost in custom_mountain/envs/hike.py obs_cost)
python ./code/main.py -p with environment.config_file=mountainHike_ver3.yaml algorithm.use_particle_filter=True log.filename='temp/'

To run Sepsis with POMCP/MCTS
Empirical model built from 1M random interactions is saved in "./POMDPy/examples/sepsis/model_256.obj"

1. POMCP (can change cost to any <= 0 value, init_idx specifies the start patient state)
python pomcp.py --init_idx 256 --cost -0.1 --is_mdp 0

2. MCTS
python pomcp.py --init_idx 256 --cost -0.05 --is_mdp 1
