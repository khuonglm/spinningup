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
        act = activation if i < sizes - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(layers)

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
            logp_a = 