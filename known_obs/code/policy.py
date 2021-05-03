import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

"""
(ex) original action space has |A| = 3 x {observe, not obs} gives
    new action space |A'| = 6 discrete actions and now prob mass is over 6 choices
- separate means first choosing between {obs, not obs} then choosing a \in original A
- learning a shared action/transition model happens in pf_model.py
"""
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, separate=False):
        super(Categorical, self).__init__()
        self.separate = separate
        if separate: # only output probs between 2 actions [push left, right]
            num_outputs = num_outputs // 2 + 1
        self.linear = nn.Linear(num_inputs, num_outputs)
        # self.p_obs = nn.Linear(num_inputs, 1)
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)
        if self.separate:
            last_x = x[:,-1]
            x = x[:,:-1]
        probs = F.softmax(x, dim=-1)
        if deterministic is False:
            action = probs.multinomial(1)
        else:
            action = probs.max(1, keepdim=True)[1]
        
        ############################################################ 
        # Can ignore this part since actions come from multinomial #
        ############################################################
        if self.separate:
            tmp = self.sigmoid(last_x)
            obs_mask = Variable(tmp > 0.5).float().unsqueeze(1)
            # action should match the env action dim
            action = action + (obs_mask == 0) * 2
        ############################################################
        
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        if self.separate:
            last_x = x[:,-1]
            x = x[:,:-1]
            p_obs, logp_obs = self.sigmoid(x), self.log_sigmoid(x)
            p_not_obs = torch.ones(p_obs.size()).to(p_obs.device) - p_obs
            logp_not_obs = p_not_obs.log()

        log_probs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)

        ############################################################ 
        # Add observation probs to action log probs & dist entropy #
        ############################################################
        if self.separate:
            batch_size, _ = actions.size()
            obs_log_probs = []
            tmp_acts = []
            for i in range(batch_size):
                if actions[i] < 2:
                    obs_log_probs.append(logp_obs[i])
                    tmp_acts.append(actions[i])
                else:
                    obs_log_probs.append(logp_not_obs[i])
                    tmp_acts.append(actions[i] - 2)
            obs_log_probs = torch.stack(obs_log_probs).to(actions.device)
            actions = torch.stack(tmp_acts).to(obs_log_probs.device)
        
        ############################################################ 
        # Use separate = False to get log probs from multinomial   #
        ############################################################
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        ############################################################
        if self.separate:
            action_log_probs += (obs_log_probs).sum(-1, keepdim=True)
            dist_entropy += -(p_obs * logp_obs + p_not_obs * logp_not_obs).sum(-1).mean()
        return action_log_probs, dist_entropy

"""
Input: Observation (latent encoding)
Output: 3-dim actions (dx, dy, obs)
"""
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, obs=True):
        super(DiagGaussian, self).__init__()
        self._obs = obs
        obs_action_dim = 3 # (dx, dy) for action, obs mask (0 or 1)
        if obs:
            self.fc_mean = nn.Linear(num_inputs, obs_action_dim)
            self.num_outputs = obs_action_dim - 1
            self.logstd = AddBias(torch.zeros(self.num_outputs))
        else:
            self.fc_mean = nn.Linear(num_inputs, num_outputs)
            self.num_outputs = num_outputs
            self.logstd = AddBias(torch.zeros(num_outputs))
        self.sigmoid = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        # zeros = torch.zeros(action_mean.size()).to(x.device)
        zeros = torch.zeros((action_mean.size()[0], self.num_outputs)).to(x.device)
        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd

    '''
    Output: 3-dim vector of (sampled) dx, dy, (deterministic) obs mask of 0 or 1
    '''
    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)
        action_std = action_logstd.exp()
        if self._obs:
            tmp = self.sigmoid(action_mean[:,-1])
            # if larger than 0.5, set observe = 1
            obs_mask = Variable(tmp > 0.5).float()
            action_mean = action_mean[:,:-1]
        
        # Returns (dx, dy) from gaussian parameterized policy network
        if deterministic is False:
            noise = torch.randn(action_std.size()).to(action_std.device)
            action = action_mean + action_std * noise
        else:
            action = action_mean
        if self._obs:
            obs_mask = obs_mask.unsqueeze(1)
            action = torch.cat((action, obs_mask), -1)
        return action

    '''
    Input: observation x, actions from self.sample
    Output: action log probs, entropy (sum of discrete and continuous action entropies)
    '''
    def logprobs_and_entropy(self, x, actions):
        batch_size, _ = actions.size()
        action_mean, action_logstd = self(x)
        action_std = action_logstd.exp()
        
        ############################################################
        #  Find probs for observing and not observing per particle #
        ############################################################
        if self._obs:
            logit = action_mean[:, -1]
            action_mean = action_mean[:, :-1]
            p_obs, logp_obs = self.sigmoid(logit), self.log_sigmoid(logit)
            p_not_obs = torch.ones(logit.size()).to(logit.device) - p_obs
            logp_not_obs = p_not_obs.log()
            
            obs_log_probs = []
            for i in range(batch_size):
                if actions[i,-1] > 0:
                    obs_log_probs.append(logp_obs[i])
                else:
                    obs_log_probs.append(logp_not_obs[i])
            obs_log_probs = torch.stack(obs_log_probs).unsqueeze(1).to(x.device)
        ############################################################

        if action_mean.shape[-1] != actions.shape[-1]:
            actions = actions[:,:-1]
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        if self._obs:
            action_log_probs = (action_log_probs + obs_log_probs).sum(-1, keepdim=True)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd 
        obs_dist_entropy = -(p_obs * logp_obs + p_not_obs * logp_not_obs).sum(-1).mean()
        dist_entropy = dist_entropy.sum(-1).mean()
        
        # log_p(action, obs) = log_p(action) + log_p(obs)
        # H(action,obs) = H(action)+H(obs)
        return action_log_probs, dist_entropy + obs_dist_entropy
