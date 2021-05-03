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

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)
        # print("x")
        # print(x)
        probs = F.softmax(x, dim=-1)
        # print("softmaxed x")
        # print(probs)
        # print("===")
        if deterministic is False:
            action = probs.multinomial(1)
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy

'''
If (dx, dy) is independent of observing, use 3 dim (dx, dy, obs=0,1),
otherwise use (dx, dy)_obs + (dx, dy)_not obs + obs={0,1}
'''
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, obs=True, cond_on_obs=False):
        super(DiagGaussian, self).__init__()
        self._obs = obs
        obs_action_dim = 3
        self.cond_on_obs = cond_on_obs
        if obs:
            # obs_action_dim = 3 if (dx, dy, o) independent
            self.fc_mean = nn.Linear(num_inputs, obs_action_dim)
            self.num_outputs = obs_action_dim-1 # 2
            self.logstd = AddBias(torch.zeros(self.num_outputs))
        else:
            self.fc_mean = nn.Linear(num_inputs, num_outputs)
            self.num_outputs = num_outputs
            self.logstd = AddBias(torch.zeros(num_outputs))
        if self.cond_on_obs:
            self.num_outputs = 2
            self.p_obs = nn.Linear(num_inputs, 1)
            self.fc_mean = nn.Linear(num_inputs+1, self.num_outputs)
            self.logstd = AddBias(torch.zeros(self.num_outputs))
        self.sigmoid = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid() # only use for getting obs log p

    def forward(self, x):
        if self.cond_on_obs:
            tmp = self.sigmoid(self.p_obs(x))
            # rand = torch.rand(tmp.size()).to(tmp.device)
            # obs_mask = Variable(rand < tmp).float()
            obs_mask = Variable(tmp[:,-1] > 0.5).float().unsqueeze(1)
            inp = torch.cat((obs_mask, x), dim=1)
            mean = self.fc_mean(inp)
            action_mean = torch.cat((mean, obs_mask), dim=1)
        else:
            action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros((action_mean.size()[0], self.num_outputs)).to(x.device)
        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)
        action_std = action_logstd.exp()

        if self._obs:
            if self.cond_on_obs: # already applied sigmoid & returns mask
                obs_mask = action_mean[:,-1]
            else:
                tmp = self.sigmoid(action_mean[:,-1])
                # if deterministic:
                obs_mask = Variable(tmp > 0.5).float()
                # else:
                #    rand = torch.rand(tmp.size()).to(tmp.device)
                #    obs_mask = Variable(rand < tmp).float()
            action_mean = action_mean[:,:-1]
            if self.num_outputs == 2:
                pass
            else: # if (dx, dy) depend on last action dim 'o'
                batch_size = obs_mask.shape[0]
                _mu = []
                _std = []
                for i in range(batch_size):
                    if obs_mask[i] > 0: # observe
                        _mu.append(action_mean[i][2:])
                        _std.append(action_std[i][2:])
                    else:
                        _mu.append(action_mean[i][:2])
                        _std.append(action_std[i][:2])
                action_mean = torch.stack(_mu).to(obs_mask.device)
                action_std = torch.stack(_std).to(obs_mask.device)

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
    Actions come from policy sample()
    '''
    def logprobs_and_entropy(self, x, actions):
        batch_size, _ = actions.size()
        if self.cond_on_obs:
            p_obs = self.sigmoid(self.p_obs(x))
            p_not_obs = torch.ones(p_obs.size()).to(x.device) - p_obs
            logp_obs, logp_not_obs = p_obs.log(), p_not_obs.log()
            obs_log_probs = []
            for i in range(batch_size):
                if actions[i,-1] > 0:
                    obs_log_probs.append(logp_obs[i])
                else:
                    obs_log_probs.append(logp_not_obs[i])
            action_mean = self.fc_mean(torch.cat((actions[:,-1].unsqueeze(1), x), dim=1))
            zeros = torch.zeros((action_mean.size()[0], self.num_outputs)).to(x.device)
            action_logstd = self.logstd(zeros)
            action_std = action_logstd.exp()
            obs_log_probs = torch.stack(obs_log_probs).to(x.device)
        ############################################################
        else:
            action_mean, action_logstd = self(x)
            logit = action_mean[:,-1]
            action_std = action_logstd.exp()
            action_mean = action_mean[:,:-1]
            p_obs, logp_obs = self.sigmoid(logit), self.log_sigmoid(logit)
            p_not_obs = torch.ones(logit.size()).to(logit.device) - p_obs
            logp_not_obs = p_not_obs.log()
            obs_log_probs = []
            dependent = self.num_outputs > 2
            if dependent:
                _mu = []
                _std = []
            for i in range(batch_size):
                if actions[i,-1] > 0: # observe
                    obs_log_probs.append(logp_obs[i])
                    if dependent:
                        _mu.append(action_mean[i][2:])
                        _std.append(action_std[i][2:])
                else:
                    obs_log_probs.append(logp_not_obs[i])
                    if dependent:
                        _mu.append(action_mean[i][:2])
                        _std.append(action_std[i][:2])
            if dependent:
                action_mean = torch.stack(_mu).to(action_mean.device)
                action_std = torch.stack(_std).to(action_mean.device)
                action_logstd = action_std.log()
            obs_log_probs = torch.stack(obs_log_probs).unsqueeze(1).to(x.device)
        ############################################################
        if action_mean.shape[-1] != actions.shape[-1]:
            actions = actions[:,:-1]
        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        # log_p(a,o|s) = {log_p(a|s) or log_p(a|o,s)} + log_p(o|s) take out obs_log_probs
        action_log_probs = (action_log_probs + obs_log_probs).sum(-1, keepdim=True)
        # H(a|o)=H(a)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd 
        obs_dist_entropy = -(p_obs * logp_obs + p_not_obs * logp_not_obs).sum(-1).mean()
        dist_entropy = dist_entropy.sum(-1).mean()
        return action_log_probs, dist_entropy + obs_dist_entropy
