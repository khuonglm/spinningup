import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_sizes=[64, 64]):
        super(Actor, self).__init__()
        
        self.pi = mlp(
            [state_size] + hidden_sizes + [action_size], 
            nn.ReLU
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.pi(state)
        action_probs = self.softmax(x)
        return action_probs      
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()
    
    def act(self, state):
        return self.get_det_action(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_sizes=[64, 64], seed=0):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.q_function = mlp(
            [state_size] + hidden_sizes + [action_size],
            nn.ReLU
        )
        for layer in self.q_function.children():
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(*hidden_init(layer))

    def forward(self, state):
        return self.q_function(state)