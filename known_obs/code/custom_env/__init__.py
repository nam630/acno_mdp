from gym.envs.registration import register

register(
        id='custom-cartpole-v0',
        entry_point='custom_env.envs:CustomCartPoleEnv',
        )
