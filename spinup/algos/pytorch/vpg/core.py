import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def combine_shape(length, shape=None):
    """
    length: scalar
    shape: tuple
    return the combined shape of [length] * [shape(n0, n1, .., nk)]
    -> tuple (length, n0, n1, ..., nk)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape) 

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Multilayer Perceptron: a type of feedforward artificial neural network 
    that consists of fully connected neurons with a nonlinear activation function, 
    organized in at least three layers
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    """
    return number of parameters of the module (model)
    """
    return np.sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(rewards, gamma):
    """
    calculate the discounted cumulative sum of rewards with 
    discount factor gamma.
    input: [r0, r1, ..., rT] 
    output: [r0 + gamma * r1 + ... + gamma^T * rT,
             r1 + gamma * r2 + ... + gamma^(T-1) * rT,
             ...,
             rT]
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        """
        return a distribution of actions from observation obs
        """
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        """
        return log_prob of an action act under a distribution pi
        """
        raise NotImplementedError
    
    def forward(self, obs, act=None):
        """
        return a distribution from action and optionally return
        the log_prob of an action act from observation obs.
        """
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    """
    MLP for Discrete environment (actions and states)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + hidden_sizes + [act_dim], activation)
    
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    """
    MLP for Continuous environment (not Discrete)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.mu_net = mlp([obs_dim] + hidden_sizes + [act_dim], activation)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    """
    MLP for state-value function
    """
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        # question about .shape[0] and .n
        # policy pi nn
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # value function v nn
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            act = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, act)
            v = self.v(obs)
        return act.numpy(), v.numpy(), logp_a.numpy()
    
    def act(self, obs):
        return self.step(obs)[0]