# import numpy as np
# import torch
# import torch.nn as nn
# from gymnasium.spaces import Box, Discrete
# # from gym.spaces import Box, Discrete
# from torch.distributions.categorical import Categorical
# from torch.distributions.kl import kl_divergence
# from torch.distributions.normal import Normal
# from torch.nn.utils import parameters_to_vector


# def count_vars(module):
#     return sum(p.numel() for p in module.parameters() if p.requires_grad)


# def keys_as_sorted_list(dict):
#     return sorted(list(dict.keys()))


# def values_as_sorted_list(dict):
#     return [dict[k] for k in keys_as_sorted_list(dict)]


# def flat_grad(f, param, **kwargs):
#     return parameters_to_vector(torch.autograd.grad(f, param, **kwargs))


# def hessian_vector_product(f, policy, x):
#     # for H = grad**2 f, compute Hx
#     g = flat_grad(f, policy.parameters(), create_graph=True)
#     return flat_grad((g * x.detach()).sum(), policy.parameters(), retain_graph=True)


# class MLP(nn.Module):
#     def __init__(
#         self,
#         layers,
#         activation=torch.tanh,
#         output_activation=None,
#         output_squeeze=False,
#     ):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#         self.activation = activation
#         self.output_activation = output_activation
#         self.output_squeeze = output_squeeze

#         for i, layer in enumerate(layers[1:]):
#             self.layers.append(nn.Linear(layers[i], layer))
#             nn.init.zeros_(self.layers[i].bias)

#     def forward(self, input):
#         x = input
#         for layer in self.layers[:-1]:
#             x = self.activation(layer(x))
#         if self.output_activation is None:
#             x = self.layers[-1](x)
#         else:
#             x = self.output_activation(self.layers[-1](x))
#         return x.squeeze() if self.output_squeeze else x


# class CategoricalPolicy(nn.Module):
#     def __init__(
#         self, in_features, hidden_sizes, activation, output_activation, action_dim
#     ):
#         super(CategoricalPolicy, self).__init__()

#         self.logits = MLP(
#             layers=[in_features] + list(hidden_sizes) + [action_dim],
#             activation=activation,
#         )

#     def forward(self, x, a=None, old_logits=None):
#         logits = self.logits(x)
#         policy = Categorical(logits=logits)

#         pi = policy.sample()
#         logp_pi = policy.log_prob(pi).squeeze()

#         if a is not None:
#             logp = policy.log_prob(a).squeeze()
#         else:
#             logp = None

#         if old_logits is not None:
#             old_policy = Categorical(logits=old_logits)
#             d_kl = kl_divergence(old_policy, policy).mean()
#         else:
#             d_kl = None

#         info = {"old_logits": logits.detach().numpy()}

#         return pi, logp, logp_pi, info, d_kl


# class GaussianPolicy(nn.Module):
#     def __init__(
#         self, in_features, hidden_sizes, activation, output_activation, action_dim
#     ):
#         super(GaussianPolicy, self).__init__()

#         self.mu = MLP(
#             layers=[in_features] + list(hidden_sizes) + [action_dim],
#             activation=activation,
#             output_activation=output_activation,
#         )
#         self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

#     def forward(self, x, a=None, old_log_std=None, old_mu=None):
#         mu = self.mu(x)
#         policy = Normal(mu, self.log_std.exp())
#         pi = policy.sample()
#         logp_pi = policy.log_prob(pi).sum(dim=1)
#         if a is not None:
#             logp = policy.log_prob(a).sum(dim=1)
#         else:
#             logp = None

#         if (old_mu is not None) or (old_log_std is not None):
#             old_policy = Normal(old_mu, old_log_std.exp())
#             d_kl = kl_divergence(old_policy, policy).mean()
#         else:
#             d_kl = None

#         info = {
#             "old_mu": np.squeeze(mu.detach().numpy()),
#             "old_log_std": self.log_std.detach().numpy(),
#         }

#         return pi, logp, logp_pi, info, d_kl


# class ActorCritic(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         action_space,
#         hidden_sizes=(64, 64),
#         activation=torch.tanh,
#         output_activation=None,
#         policy=None,
#     ):
#         super(ActorCritic, self).__init__()

#         if policy is None and isinstance(action_space, Box):
#             self.policy = GaussianPolicy(
#                 in_features,
#                 hidden_sizes,
#                 activation,
#                 output_activation,
#                 action_dim=action_space.shape[0],
#             )
#         elif policy is None and isinstance(action_space, Discrete):
#             self.policy = CategoricalPolicy(
#                 in_features,
#                 hidden_sizes,
#                 activation,
#                 output_activation,
#                 action_dim=action_space.n,
#             )
#         else:
#             self.policy = policy(
#                 in_features, hidden_sizes, activation, output_activation, action_space
#             )

#         self.value_function = MLP(
#             layers=[in_features] + list(hidden_sizes) + [1],
#             activation=activation,
#             output_squeeze=True,
#         )

#     def forward(self, x, a=None, **kwargs):
#         pi, logp, logp_pi, info, d_kl = self.policy(x, a, **kwargs)
#         v = self.value_function(x)

#         return pi, logp, logp_pi, info, d_kl, v

#     def act(self, obs):
#         return self.forward(obs)[0].detach().numpy()

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


# TRPO utilities
def flat_grads(grads):
    return torch.cat([grad.contiguous().view(-1) for grad in grads])


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]