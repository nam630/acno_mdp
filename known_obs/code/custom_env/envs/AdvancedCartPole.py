import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from collections import deque
import torch
import copy


class CustomCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees (anything past 12 degrees is considered critical/deceased)
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Solved Requirements
        Considered solved when the pole has been kept at angle between 6 degrees for over 100 consecutive steps
        (anything less than 6 is considered healthy, must stay in the healthy condition for 100 consecutive steps before terminating successfully)
        Any angle between 6 and 12 is considered volatile
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    '''
    In the recent version (02/01), use the original reward function & same termination condition
    Reward +1 for every step, max episode length 200, terminates early if the pole falls behind the threshold
    Goal is to keep the pole standing as long as possible in the 200 step episode.
    In the changed version, there's no distinction between surviving for 99 steps and failing at step 0
    '''

    def __init__(self,
                obs_cost=-0.75, 
                noise=False,
                per_step_reward=False,
                counter=False,
                no_missingness=False,
                action_aug=False,
                ):
        self.no_missingness = no_missingness # no missing observation; 2 of the action choices correspond to the same control action
        self.cost = obs_cost
        self.max_episode_step = 200
        self.final_rew = 0 # -1500
        self.total_rew = 0
        self.obs_num = 0
        self.success = 0 # number of consecutively successful steps
        self.steps = 0
        self.noise = noise
        self.action_aug = action_aug
        ## TODO : adding noise is not implemented yet
        if noise:
            self.perturb_rate = 0.03
        self.counter = counter
        self.per_step_reward = per_step_reward # per step reward could potentially be misleading since -1 is still given for every success state until the agent has reached 100 consecutive steps
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle at which to keep the agent to be considered stable
        self.theta_success_threshold_radians = 6 * 2 * math.pi / 360

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        actions_n = 2
        if not no_missingness:
            actions_n *= 2
        # before spaces.Discrete(2) probably meant the action was not actually sampled properly
        self.action_space = spaces.Discrete(actions_n)

        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        self.obs = None
        # only for debugging
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """
    if observe = True, observe the current state
    if action = 0, push Left;
    action = 1, push Right
    """
    def _decouple(self, action):
        observe = False
        motion = 1
        if action < 2:
            observe = True
        if action % 2 == 0: # 0, 1, 2, 3
            motion = 0
        return observe, motion

    def step(self, combined_action):
        ###### OVERWRITE action to always be random ######
        observe, action = self._decouple(combined_action)
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)
        
        fell =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        fell = bool(fell)
        reward = 1
        self.steps += 1
        if fell:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
                reward += self.final_rew
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                assert(False)
        
        success = theta < self.theta_success_threshold_radians \
                and theta > - self.theta_success_threshold_radians
        success = bool(success)
        if success:
            # For every consecutive step spent in the "success" zone
            self.success += 1
        else:
            self.success = 0

        # terminate_success = bool(self.success >= 100) 
        # if self.per_step_reward:
        #    reward -= 1
       
        # If no missingness, always update the observation with the true state
        if self.no_missingness:
            self.obs[:4] = self.state
            if self.counter:
                self.obs[4] = 0
        else:   
            if observe:
                self.obs[:4] = self.state
                if self.counter:
                    self.obs[4] = 0
                    if self.action_aug:
                        self.obs[5] = action 
                reward += self.cost
                self.obs_num += 1
            # only if didn't observe
            else:
                # impute with NULL (all zeros)
                self.obs[:4] = np.zeros((4,), dtype=float)
                
                # uncomment for last observation carry forward
                # self.obs[:4] = self.obs[:4]
                
                # increment the counter if unobserved
                if self.counter:
                    self.obs[4] = self.obs[4] + 1
                    if self.action_aug:
                        self.obs[5] = action

        done = bool(self.steps == self.max_episode_step or fell)
        self.total_rew += reward

        return np.array(self.obs).copy(), reward, done, {'action': action, 'success': self.steps == self.max_episode_step, 'comb action': combined_action, 'observed': observe, 'true state': np.array(self.state), 'total rew': self.total_rew, 'fell': fell, 'obs_num':self.obs_num, 'steps': self.steps}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.obs = np.array(self.state)
        if self.counter:
            self.obs = np.concatenate((self.obs, [0]))
            if self.action_aug:
                # to indicate no prev action
                self.obs = np.concatenate((self.obs, [-1]))
        self.steps = 0
        self.success = 0
        self.obs_num = 0
        self.total_rew = 0.
        return np.array(self.obs)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
