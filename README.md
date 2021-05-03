# acno_mdp
# Conda packages and versions used for generating the reported results are shared in conda.yml (Note not all the packages are needed).
# To run the known observation belief encoder

# cartpole (can adjust observation cost in cofig_file)
python known_obs/code/main_known.py -p with environment.config_file=cartpole_ver3.yaml
# mountain hike (can adjust observation cost in config file)
python known_obs/code/main_known.py -p with environment.config_file=mountainHike_ver2.yaml

# To run the default DVRL belief encoder (also need to manually set env_id in code/envs.py 'make_env' -- review inline comments)

# cartpole (need to set obs_cost in custom_cartpole/envs/AdvancedCartPole.py obs_cost)
python ./code/main.py -p with environment.config_file=cartpole_ver2.yaml algorithm.use_particle_filter=True log.filename='temp/'

# mountain (need to set obs_cost in custom_mountain/envs/hike.py obs_cost)
python ./code/main.py -p with environment.config_file=mountainHike_ver3.yaml algorithm.use_particle_filter=True log.filename='temp/'


