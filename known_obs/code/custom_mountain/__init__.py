from gym.envs.registration import register

register(
        id='hike-v1',
        entry_point='custom_mountain.envs:CustomHikeEnv',
        )


