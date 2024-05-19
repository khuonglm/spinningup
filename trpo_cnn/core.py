import numpy as np
import torch
import torch.nn as nn
# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium.spaces import Box, Discrete
import gym
from gym import spaces
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.utils import parameters_to_vector

def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def flat_grad(f, param, **kwargs):
    return parameters_to_vector(torch.autograd.grad(f, param, **kwargs))


def hessian_vector_product(f, policy, x):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, policy.parameters(), create_graph=True)
    return flat_grad((g * x.detach()).sum(), policy.parameters(), retain_graph=True)


def transpose(obs):
    """
    obs with shape (B, H, W, C) -> (B, C, H, W)
                    (H, W, C) -> (C, H, W)
    """
    if isinstance(obs, torch.Tensor):
        obs = obs.numpy()
    else:
        obs = np.asarray(obs, dtype=np.float32)
    x = torch.from_numpy(np.moveaxis(obs, -1, len(obs.shape) == 4))
    return x

class MLP(nn.Module):
    def __init__(
        self,
        layers,
        activation=torch.tanh,
        output_activation=None,
        output_squeeze=False,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):
    def __init__(
        self, in_features, hidden_sizes, activation, output_activation, action_dim
    ):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
        )

    def forward(self, x, a=None, old_logits=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)

        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        if old_logits is not None:
            old_policy = Categorical(logits=old_logits)
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {"old_logits": logits.detach().numpy()}

        return pi, logp, logp_pi, info, d_kl


class GaussianPolicy(nn.Module):
    def __init__(
        self, in_features, hidden_sizes, activation, output_activation, action_dim
    ):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
            output_activation=output_activation,
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, x, a=None, old_log_std=None, old_mu=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        if (old_mu is not None) or (old_log_std is not None):
            old_policy = Normal(old_mu, old_log_std.exp())
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {
            "old_mu": np.squeeze(mu.detach().numpy()),
            "old_log_std": self.log_std.detach().numpy(),
        }

        return pi, logp, logp_pi, info, d_kl


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_features,
        action_space,
        hidden_sizes=(64, 64),
        activation=torch.tanh,
        output_activation=None,
        policy=None,
    ):
        super(ActorCritic, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.shape[0],
            )
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.n,
            )
        else:
            self.policy = policy(
                in_features, hidden_sizes, activation, output_activation, action_space
            )

        self.value_function = MLP(
            layers=[in_features] + list(hidden_sizes) + [1],
            activation=activation,
            output_squeeze=True,
        )

    def forward(self, x, a=None, **kwargs):
        pi, logp, logp_pi, info, d_kl = self.policy(x, a, **kwargs)
        v = self.value_function(x)

        return pi, logp, logp_pi, info, d_kl, v
    
    def act(self, obs):
        return self.forward(obs)[0].detach().numpy()

class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param in_features: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        in_features: int = 512,
        layers=[],
        activation=torch.tanh,
        output_activation=None,
        output_squeeze=False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        assert in_features > 0
        super(NatureCNN, self).__init__()
        self._observation_space = observation_space
        self._in_features = in_features
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                transpose(torch.as_tensor(observation_space.sample()).float())
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, in_features), nn.ReLU())

        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    @property
    def in_features(self) -> int:
        return self._in_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.linear(self.cnn(observations / 255.0))
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CNNCategoricalPolicy(nn.Module):
    def __init__(
        self, 
        observation_space, 
        hidden_sizes, 
        activation,
        in_features, 
        output_activation, 
        action_dim
    ):
        super(CNNCategoricalPolicy, self).__init__()
        self.logits = NatureCNN(
            observation_space=observation_space,
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
            in_features=in_features,
            output_activation=output_activation,
        )

    def forward(self, x, a=None, old_logits=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)

        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        if old_logits is not None:
            old_policy = Categorical(logits=old_logits)
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {"old_logits": logits.detach().numpy()}

        return pi, logp, logp_pi, info, d_kl


class CNNGaussianPolicy(nn.Module):
    def __init__(
        self, 
        observation_space, 
        hidden_sizes, 
        activation,
        in_features, 
        output_activation, 
        action_dim
    ):
        super(CNNGaussianPolicy, self).__init__()

        self.mu = NatureCNN(
            observation_space=observation_space,
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
            in_features=in_features,
            output_activation=output_activation,
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, x, a=None, old_log_std=None, old_mu=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        if (old_mu is not None) or (old_log_std is not None):
            old_policy = Normal(old_mu, old_log_std.exp())
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {
            "old_mu": np.squeeze(mu.detach().numpy()),
            "old_log_std": self.log_std.detach().numpy(),
        }

        return pi, logp, logp_pi, info, d_kl


class CNNActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        in_features=512,
        hidden_sizes=(64, 64),
        activation=torch.tanh,
        output_activation=None,
        policy=None,
    ):
        super(CNNActorCritic, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = CNNGaussianPolicy(
                observation_space=observation_space,
                in_features=in_features,
                hidden_sizes=hidden_sizes,
                activation=activation,
                output_activation=output_activation,
                action_dim=action_space.shape[0],
            )
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CNNCategoricalPolicy(
                observation_space=observation_space,
                in_features=in_features,
                hidden_sizes=hidden_sizes,
                activation=activation,
                output_activation=output_activation,
                action_dim=action_space.n,
            )
        else:
            self.policy = policy(
                in_features, hidden_sizes, activation, output_activation, action_space
            )

        self.value_function = NatureCNN(
            observation_space=observation_space,
            in_features=in_features,
            layers=[in_features] + list(hidden_sizes) + [1],
            activation=activation,
            output_squeeze=True,
        )

    def forward(self, x, a=None, **kwargs):
        x = transpose(x)
        pi, logp, logp_pi, info, d_kl = self.policy(x, a, **kwargs)
        v = self.value_function(x)

        return pi, logp, logp_pi, info, d_kl, v
    
    def act(self, obs):
        return self.forward(obs)[0].detach().numpy()