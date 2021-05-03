import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import aesmc.random_variable as rv
# import aesmc.autoencoder as ae
import aesmc.state as st
import aesmc.util as ae_util
import aesmc.statistics as stats
import aesmc.math as math
import aesmc.test_utils as tu
from aesmc.inference import sample_ancestral_index
import encoder_decoder
import numpy as np
import model
from operator import mul
from functools import reduce

import h5py

class PF_State():
    def __init__(self, particle_state, particle_log_weights):
        self.particle_state = particle_state
        self.particle_log_weights = particle_log_weights

    def detach(self):
        return PF_State(
            self.particle_state.detach(),
            self.particle_log_weights.detach())

    def cuda(self):
        return PF_State(
            self.particle_state.cuda(),
            self.particle_log_weights.cuda())


class DVRLPolicy(model.Policy):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 observation_type,
                 action_encoding,
                 # obs_encoding,
                 cnn_channels,
                 h_dim,
                 init_function,
                 encoder_batch_norm,
                 policy_batch_norm,
                 prior_loss_coef,
                 obs_loss_coef,
                 detach_encoder,
                 batch_size,
                 num_particles,
                 particle_aggregation,
                 z_dim,
                 resample,
                 savedir,
                 env_type='mountain', # choose between 'cartpole', 'mountain'
                 verbose=False,
                 ):
        super().__init__(action_space, encoding_dimension=h_dim)
        self.init_function = init_function
        self.num_particles = num_particles
        self.particle_aggregation = particle_aggregation
        self.batch_size = batch_size
        self.obs_loss_coef = float(obs_loss_coef)
        self.prior_loss_coef = float(prior_loss_coef)
        self.observation_type = observation_type
        self.encoder_batch_norm = encoder_batch_norm
        self.policy_batch_norm = policy_batch_norm
        self.detach_encoder = detach_encoder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.resample = resample
        self.env = env_type
           
        self.halved_acts = False # always keep false
        if self.env == 'cartpole':
            if self.halved_acts:
                nr_actions=2
            else:
                nr_actions=4
            enc_actions = nr_actions
        elif self.env == 'mountain':
            nr_actions = 3
            enc_actions = nr_actions
            if self.halved_acts: # know only 2 actions (dx, dy) affect the agent dynamics
                enc_actions -= 1
                
        self.action_encoding=128
        self.action_encoder = nn.Sequential(
                nn.Linear(enc_actions, action_encoding),
                nn.ReLU()
                )
        self.nr_actions = nr_actions

        # 10 or 30 for particle sz
        batch_sz = 16
        if self.env == 'cartpole':
            state_dim = 4
        elif self.env =='mountain':
            state_dim = 2
        
        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
        else:
            action_shape = action_space.shape[0]

        # Computes p(z_t|h_{t-1}, a_{t-1})
        self.transition_network = VRNN_transition(
            h_dim=h_dim,
            z_dim=z_dim,
            action_encoding=action_encoding
            )

        # Encode the observation
        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
                observation_type,
                cnn_channels)
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)
        phi_x_dim = self.cnn_output_number
        self.phi_x = encoder_decoder.get_encoder(
            observation_type,
            nr_inputs,
            cnn_channels,
            batch_norm=encoder_batch_norm,
            )
        
        # Computes h_t=f(h_{t-1}, z_t, a_{t-1}, o_t)
        self.deterministic_transition_network = VRNN_deterministic_transition(
            z_dim=z_dim,
            phi_x_dim=phi_x_dim,
            h_dim=h_dim,
            action_encoding=action_encoding
            )

        # Only for aggregating belief particles into a 1-d input for linear actor critic
        dim = 2 * h_dim + 1 # h_dim = z_dim, + 1 (for each particle weight)
        if particle_aggregation == 'rnn' and self.num_particles > 1:
            self.particle_gru = nn.GRU(dim, h_dim, batch_first=True)

        elif self.num_particles == 1:
            self.particle_gru = nn.Linear(dim, h_dim*2)

        self.reset_parameters()

    def new_latent_state(self):
        """
        Return new latent state.
        This is a function because the latent state is different for DVRL and RNN.
        """
        device = next(self.parameters()).device
        initial_state = st.State(
            h=torch.zeros(self.batch_size, self.num_particles, self.h_dim).to(device))

        log_weight = torch.zeros(self.batch_size, self.num_particles).to(device)

        initial_state.log_weight = log_weight

        return initial_state

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        Set latent state to 0-tensors when new episode begins.
        Args:
            latent_state (`State`): latent_state
            mask: binary tensor with 0 whenever a new episode begins.

        """
        # Multiply log_weight, h, z with mask
        return latent_state.multiply_each(mask, only=['log_weight', 'h', 'z'])

    def reset_parameters(self):
        def weights_init(gain):
            def fn(m):
                classname = m.__class__.__name__
                init_func = getattr(torch.nn.init, self.init_function)
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    init_func(m.weight.data, gain=gain)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                if classname.find('GRUCell') != -1:
                    init_func(m.weight_ih.data)
                    init_func(m.weight_hh.data)
                    m.bias_ih.data.fill_(0)
                    m.bias_hh.data.fill_(0)

            return fn

        relu_gain = nn.init.calculate_gain('relu')
        self.apply(weights_init(relu_gain))
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def sample_from(self, state_random_variable):
        """
        Helper function, legazy code.
        """
        return state_random_variable.sample_reparameterized(
            self.batch_size, self.num_particles
        )

    def encode(self, observation, reward, actions, previous_latent_state, predicted_times, savedir=None, verbose=True):
        """
        This is where the core of the DVRL algorithm is happening.
        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State
            predicted_times (list of ints): List of timesteps into the future for which predictions
                                            should be returned. Only makes sense if
                                            encoding_loss_coef != 0 and obs_loss_coef != 0

        return latent_state, \
            - encoding_logli, \
            (- transition_logpdf + proposal_logpdf, - emission_logpdf),\
            avg_num_killed_particles,\
            predicted_observations, particle_observations
        Returns:
            latent_state: New latent state
            - transition_logpdf 
            predicted_particles: List of Nones
        """
        batch_size, *rest = observation.size()

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)
        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        actions = actions.unsqueeze(0)
        reward = reward.unsqueeze(0)
        
        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_a=actions.contiguous(),
            r=reward.contiguous()
        )

        old_log_weight = previous_latent_state.log_weight
        # Linearly connected layer to expand one hot encoded action to 128
        seq_len, batch_size, *obs_dim = observation_states.all_x.size()
        if self.action_encoding > 0:
            if self.env == 'cartpole':
                # reduce the action space for transition model by half
                if self.halved_acts:
                    all_acts = []
                    sample_sz, _, _ = observation_states.all_a.shape
                    for b in range(batch_size):
                        act = observation_states.all_a[0, b, :]
                        if sum(act[:2]).item() > 1:
                            all_acts.append(act[:2])
                        else:
                            all_acts.append(act[2:])
                    all_acts = torch.stack(all_acts).to(observation_states.all_a.device)
                    all_acts = all_acts.unsqueeze(0)
                else:
                    all_acts = observation_states.all_a.view(-1, self.nr_actions)
            elif self.env == 'mountain':
                all_acts = observation_states.all_a.view(-1, self.nr_actions)
                if self.halved_acts:
                    # only use relevant part of the actions for training the dynamics model
                    all_acts = all_acts[:,:-1]
            encoded_action = self.action_encoder(all_acts).view(seq_len, batch_size, -1)
            observation_states.encoded_action = encoded_action
        
        # Encode the observations and expand
        all_phi_x = self.phi_x(
                observation_states.all_x.view(-1, *obs_dim)
                ).view(-1, self.cnn_output_number)
        all_phi_x = all_phi_x.view(seq_len, batch_size, -1)
        observation_states.all_phi_x = all_phi_x
        
        # Expand the particle dimension
        observation_states.unsequeeze_and_expand_all_(dim=2, size=self.num_particles)
        ancestral_indices = sample_ancestral_index(old_log_weight)

        # How many particles were killed?
        # List over batch size
        num_killed_particles = list(tu.num_killed_particles(ancestral_indices.data.cpu()))
        if self.resample:
            previous_latent_state = previous_latent_state.resample(ancestral_indices)
        else:
            num_killed_particles = [0] * batch_size
        avg_num_killed_particles = sum(num_killed_particles)/len(num_killed_particles)
        
        # Legacy code: Select first (and only) time index
        current_observation = observation_states.index_elements(0)
        # Minimize the distance between true obs and predicted latent state
        transition_state_random_variable = self.transition_network(
            previous_latent_state,
            current_observation
            )
        latent_state = self.sample_from(transition_state_random_variable)
        
        '''
        Check if a true observation is obtained // needs to consider different action representations for envs
        Manually set all z's to be the observation
        '''
        for sample_i in range(batch_size):
            if self.env == 'cartpole' and sum(actions[0, sample_i, :2] == 1) > 0:
                latent_state.z[sample_i,:,:] = Variable(current_observation.all_x[sample_i,:,:].clone()).to(observation.device)
            if self.env == 'mountain' and actions[0, sample_i, -1].item() > 0:
                # might have to reshape using current_obesrvation.all_x[sample_i,:,:].squeeze(1) for eval
                latent_state.z[sample_i,:,:] = Variable(current_observation.all_x[sample_i,:,:].clone()).to(observation.device)
        
        latent_state = self.deterministic_transition_network(
                previous_latent_state=previous_latent_state,
                latent_state=latent_state,
                observation_states=current_observation,
                time=0,
                )

        # make sure to check for sign & what is accounting for
        mu = transition_state_random_variable.z._mean
        var = transition_state_random_variable.z._variance
        batch_size, num_particles, z_dim = mu.shape
        # want to maximize log likelihood of true observation under the model
        transition_logpdf = torch.sum(
                            (-0.5 * (current_observation.all_x - mu)**2 / var -\
                            0.5 * torch.log(2 * var * np.pi)\
                            ).view(batch_size, num_particles, -1), dim=2
                            )
        # to ensure lines below don't affect loss update
        new_log_weight = transition_logpdf.clone()
        
        for sample_i in range(batch_size):
            if self.env == 'cartpole':
                if sum(actions[0, sample_i, :2] == 1) > 0: # if observed
                    pass
                else: # if observations are missing, cannot compute prob
                    transition_logpdf[sample_i] *= 0
            elif self.env == 'mountain':
                if actions[0, sample_i, -1].item() > 0: # if observed
                    pass
                else:
                    transition_logpdf[sample_i] *= 0

        assert(self.prior_loss_coef == 1)
        assert(self.obs_loss_coef == 1)

        latent_state.log_weight = new_log_weight
        
        # Average (in log space) over particles
        encoding_logli = math.logsumexp(
            # torch.stack(log_weights, dim=0), dim=2
            transition_logpdf, dim=1
        ) - np.log(self.num_particles)

        predicted_observations = None
        particle_observations = None
        ae_util.init(False)
        # Hack to ignore emission loss
        emission_logpdf = torch.zeros(transition_logpdf.size()).to(transition_logpdf.device)
        return latent_state, \
            - encoding_logli, \
            (- transition_logpdf, - emission_logpdf),\
            avg_num_killed_particles,\
            predicted_observations, particle_observations


    def predict_observations(self, latent_state, current_observation, actions,
                             emission_state_random_variable, predicted_times):
        """
        Assumes that the current encoded action (saved in 'current_observation') is
        repeated into the future
        """

        max_distance = max(predicted_times)
        old_log_weight = latent_state.log_weight
        predicted_observations = []
        particle_observations = []

        if 0 in predicted_times:
            x = emission_state_random_variable.all_x._probability

            averaged_obs = stats.empirical_mean(
                x,
                old_log_weight)
            predicted_observations.append(averaged_obs)
            particle_observations.append(x)

        batch_size, num_particles, z_dim = latent_state.z.size()
        batch_size, num_particles, h_dim = latent_state.h.size()
        for dt in range(max_distance):
            old_observation = current_observation
            previous_latent_state = latent_state

            # Get next state
            transition_state_random_variable = self.transition_network(
                previous_latent_state,
                old_observation
                )
            latent_state = self.sample_from(transition_state_random_variable)

            # Hack. This is usually done in det_transition
            # latent_state.phi_z = self.deterministic_transition_network.phi_z(
            #    latent_state.z.view(-1, z_dim)).view(batch_size, num_particles, h_dim)

            # Draw observation
            emission_state_random_variable = self.emission_network(
                previous_latent_state,
                latent_state,
                old_observation
                # observation_states
                )
            x = emission_state_random_variable.all_x._probability
            averaged_obs = stats.empirical_mean(
                x,
                old_log_weight)

            # Encode observation
            # Unsqueeze time dimension
            current_observation = st.State(
                all_x=averaged_obs.unsqueeze(0),
                all_a=actions.contiguous()
            )
            current_observation = self.encoding_network(current_observation)
            current_observation.unsequeeze_and_expand_all_(dim=2, size=self.num_particles)
            current_observation = current_observation.index_elements(0)

            # Deterministic update
            latent_state = self.deterministic_transition_network(
                previous_latent_state=previous_latent_state,
                latent_state=latent_state,
                observation_states=current_observation,
                time=0
            )

            if dt+1 in predicted_times:
                predicted_observations.append(averaged_obs)
                particle_observations.append(x)

        return predicted_observations, particle_observations


    def encode_particles(self, latent_state):
        """
        RNN that encodes the set of particles into one latent vector that can be passed to policy.
        """
        batch_size, num_particles, h_dim = latent_state.h.size()
        state = torch.cat([latent_state.h,
                        latent_state.phi_z],
                        dim=2)
        normalized_log_weights = math.lognormexp(
            latent_state.log_weight,
            dim=1
        )

        particle_state = torch.cat(
            [state,
             torch.exp(normalized_log_weights).unsqueeze(-1)],
            dim=2)
        if self.num_particles == 1:
            particle_state = particle_state.squeeze(1)
            encoded_particles = self.particle_gru(particle_state)
            return encoded_particles
        else:
            _ , encoded_particles = self.particle_gru(particle_state)
            return encoded_particles[0]


class VRNN_transition(nn.Module):
    def __init__(self, h_dim, z_dim, action_encoding):
        super().__init__()
        self.prior = nn.Sequential(
            nn.Linear(h_dim + action_encoding, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, observation_states):
        """Outputs the prior probability of z_t.

        Inputs:
            - previous_latent_state containing at least
                `h`     [batch, particles, h_dim]
        """

        batch_size, num_particles, h_dim = previous_latent_state.h.size()
        
        if self.action_encoding > 0:
            input = torch.cat([
                previous_latent_state.h,
                observation_states.encoded_action
            ], 2).view(-1, h_dim + self.action_encoding)
        else:
            input = previous_latent_state.h.view(-1, h_dim)

        prior_t = self.prior(input)

        prior_mean_t = self.prior_mean(prior_t).view(batch_size, num_particles, -1)
        prior_std_t = self.prior_std(prior_t).view(batch_size, num_particles, -1)

        prior_dist = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(
                mean=prior_mean_t,
                variance=prior_std_t
                ))

        return prior_dist

class VRNN_deterministic_transition(nn.Module):
    def __init__(self, z_dim, phi_x_dim, h_dim, action_encoding):
        super().__init__()
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())
        # From phi_z and phi_x_dim
        self.rnn = nn.GRUCell(h_dim + phi_x_dim + action_encoding, h_dim)
        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, latent_state, observation_states, time):
        batch_size, num_particles, z_dim = latent_state.z.size()
        batch_size, num_particles, phi_x_dim = observation_states.all_phi_x.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        phi_x = observation_states.all_phi_x

        phi_z_t = self.phi_z(latent_state.z.view(-1, z_dim)).view(batch_size, num_particles, h_dim)

        if self.action_encoding > 0:
            input = torch.cat([
                phi_x,
                phi_z_t,
                observation_states.encoded_action
            ], 2).view(-1, phi_x_dim + h_dim + self.action_encoding)
        else:
            input = torch.cat([
                phi_x,
                phi_z_t
            ], 1).view(-1, phi_x_dim + h_dim)

        h = self.rnn(
            input,
            previous_latent_state.h.view(-1, h_dim))

        latent_state.phi_z = phi_z_t.view(batch_size, num_particles, -1)
        # We need [batch, particles, ...] for aesmc resampling!
        latent_state.h = h.view(batch_size, num_particles, h_dim)
        return latent_state


