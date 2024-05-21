# import numpy as np
# import random
# import time
# import copy
# from collections import deque, namedtuple
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.nn.utils import clip_grad_norm_
# import gymnasium as gym

# from spinup.algos.pytorch.sacd import core
# from spinup.utils.logx import EpochLogger

# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""

#     def __init__(self, buffer_size, batch_size, device):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """
#         self.device = device
#         self.memory = deque(maxlen=buffer_size)  
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
    
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.memory, k=self.batch_size)

#         states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
#         next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)


# class SAC(nn.Module):
#     """Interacts with and learns from the environment."""
    
#     def __init__(
#             self,
#             state_size,
#             action_size,
#             device,
#             hidden_sizes,
#         ):
#         """Initialize an Agent object.
        
#         Params
#         ======
#             state_size (int): dimension of each state
#             action_size (int): dimension of each action
#         """
#         super(SAC, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size

#         self.device = device
        
#         self.gamma = 0.99
#         self.tau = 1e-2
#         # hidden_size = 256
#         learning_rate = 5e-4
#         self.clip_grad_param = 1

#         self.target_entropy = -action_size  # -dim(A)

#         self.log_alpha = torch.tensor([0.0], requires_grad=True)
#         self.alpha = self.log_alpha.exp().detach()
#         self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
                
#         # Actor Network 
#         self.actor_local = core.Actor(state_size, action_size, hidden_sizes).to(device)
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
#         # Critic Network (w/ Target Network)
#         self.critic1 = core.Critic(state_size, hidden_sizes).to(device)
#         self.critic2 = core.Critic(state_size, hidden_sizes).to(device)
        
#         assert self.critic1.parameters() != self.critic2.parameters()
        
#         self.critic1_target = core.Critic(state_size, hidden_sizes).to(device)
#         self.critic1_target.load_state_dict(self.critic1.state_dict())

#         self.critic2_target = core.Critic(state_size, hidden_sizes).to(device)
#         self.critic2_target.load_state_dict(self.critic2.state_dict())

#         self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
#         self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    
#     def get_action(self, state):
#         """Returns actions for given state as per current policy."""
#         state = torch.from_numpy(state).float().to(self.device)
        
#         with torch.no_grad():
#             action = self.actor_local.get_det_action(state)
#         return action.numpy()

#     def calc_policy_loss(self, states, alpha):
#         _, action_probs, log_pis = self.actor_local.evaluate(states)

#         q1 = self.critic1(states)   
#         q2 = self.critic2(states)
#         min_Q = torch.min(q1,q2)
#         actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
#         log_action_pi = torch.sum(log_pis * action_probs, dim=1)
#         return actor_loss, log_action_pi
    
#     def learn(self, step, experiences, gamma, d=1):
#         """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
#         Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
#         Critic_loss = MSE(Q, Q_target)
#         Actor_loss = α * log_pi(a|s) - Q(s,a)
#         where:
#             actor_target(state) -> action
#             critic_target(state, action) -> Q-value
#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
#             gamma (float): discount factor
#         """
#         states, actions, rewards, next_states, dones = experiences

#         # ---------------------------- update actor ---------------------------- #
#         current_alpha = copy.deepcopy(self.alpha)
#         actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
        
#         # Compute alpha loss
#         alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
#         self.alpha_optimizer.zero_grad()
#         alpha_loss.backward()
#         self.alpha_optimizer.step()
#         self.alpha = self.log_alpha.exp().detach()

#         # ---------------------------- update critic ---------------------------- #
#         # Get predicted next-state actions and Q values from target models
#         with torch.no_grad():
#             _, action_probs, log_pis = self.actor_local.evaluate(next_states)
#             Q_target1_next = self.critic1_target(next_states)
#             Q_target2_next = self.critic2_target(next_states)
#             Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

#             # Compute Q targets for current states (y_i)
#             Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

#         # Compute critic loss
#         q1 = self.critic1(states).gather(1, actions.long())
#         q2 = self.critic2(states).gather(1, actions.long())
        
#         critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
#         critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

#         # Update critics
#         # critic 1
#         self.critic1_optimizer.zero_grad()
#         critic1_loss.backward(retain_graph=True)
#         clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
#         self.critic1_optimizer.step()
#         # critic 2
#         self.critic2_optimizer.zero_grad()
#         critic2_loss.backward()
#         clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
#         self.critic2_optimizer.step()

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.critic1, self.critic1_target)
#         self.soft_update(self.critic2, self.critic2_target)
        
#         return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

#     def soft_update(self, local_model , target_model):
#         """Soft update model parameters.
#         θ_target = τ*θ_local + (1 - τ)*θ_target
#         Params
#         ======
#             local_model: PyTorch model (weights will be copied from)
#             target_model: PyTorch model (weights will be copied to)
#             tau (float): interpolation parameter 
#         """
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


# def collect_random(env, dataset, num_samples=200):
#     state, _ = env.reset()
#     for _ in range(num_samples):
#         action = env.action_space.sample()
#         next_state, reward, done, t, _ = env.step(action)
#         done = done | t
#         dataset.add(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             state, _ = env.reset()


# def sacd(env_fn, sac=SAC, ac_kwargs=dict(), seed=0, steps_per_epoch=4000, 
#         epochs=100, replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, 
#         alpha=0.2, batch_size=100, start_steps=10000, update_after=1000, 
#         update_every=50, num_test_episodes=10, max_ep_len=1000, 
#         logger_kwargs=dict(), save_freq=1):
    
#     logger = EpochLogger(**logger_kwargs)
#     logger.save_config(locals())

#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)

#     env = env_fn()
#     # env.seed(seed)
#     # env.action_space.seed(seed)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.n
    
#     agent = SAC(state_size=obs_dim, action_size=act_dim, device=device, **ac_kwargs)

#     # save model
#     logger.setup_pytorch_saver(agent.actor_local)

#     buffer = ReplayBuffer(buffer_size=replay_size, batch_size=batch_size, device=device)
    
#     collect_random(env=env, dataset=buffer, num_samples=10000)
#     start_time = time.time()
#     steps = 0
#     max_avg_ret = 0
#     avg_ret = 0
#     count = 0

#     for i in range(epochs):
#         for t in range(steps_per_epoch):
#             state, _ = env.reset()
#             episode_steps = 0
#             rewards = 0
#             while True:
#                 action = agent.get_action(state)
#                 next_state, reward, done, t, _ = env.step(action)
#                 steps += 1
#                 done = done | t
#                 buffer.add(state, action, reward, next_state, done)
#                 policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
#                 state = next_state
#                 rewards += reward
#                 episode_steps += 1
#                 if done:
#                     break
        
#             logger.store(EpRet=rewards, EpLen=episode_steps)
#             logger.store(LossPi=policy_loss, AlphaLoss=alpha_loss)
#             logger.store(BelErr1=bellmann_error1, BelError2=bellmann_error2)
#             logger.store(CurAlpha=current_alpha)
#             avg_ret += rewards
#             count += 1

#         logger.log_tabular('Epoch', i)
#         logger.log_tabular('EpRet', with_min_and_max=True)
#         logger.log_tabular('EpLen', average_only=True)
#         logger.log_tabular('TotalEnvInteracts', (i+1)*steps_per_epoch)
#         logger.log_tabular('LossPi', average_only=True)
#         logger.log_tabular('AlphaLoss', average_only=True)
#         logger.log_tabular('BelErr1', average_only=True)
#         logger.log_tabular('BelErr2', average_only=True)
#         logger.log_tabular('CurAlpha', average_only=True)
#         logger.log_tabular('Time', time.time()-start_time)
#         logger.dump_tabular()

#         if i % save_freq == 0 or i == epochs - 1:
#             avg_ret  = float(avg_ret / count)
#             if avg_ret > max_avg_ret:
#                 logger.save_state({'env': env}, None)
#                 max_avg_ret = avg_ret
            
#             avg_ret, count = 0, 0


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='sac')
#     args = parser.parse_args()

#     from spinup.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     torch.set_num_threads(torch.get_num_threads())

#     sacd(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
#         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#         logger_kwargs=logger_kwargs)