import time
import numpy as np
import torch
from torch.optim import Adam
import gym

import spinup.algos.pytorch.ppo.core as core
from spinup.utils.mpi_tools import mpi_statistics_scalar, proc_id, num_procs, mpi_fork
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.logx import EpochLogger

class PPOBuffer:
    """
    Buffer to store experience by interactions with the environment
    """
    def __init__(self, obs_dim, act_dim, buf_size, gamma, lam):
        self.obs_buf = np.zeros(core.combine_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combine_shape(buf_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)
        self.adv_buf = np.zeros(buf_size, dtype=np.float32)
        self.val_buf = np.zeros(buf_size, dtype=np.float32)
        self.logp_buf = np.zeros(buf_size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.episode_start, self.ptr, self.max_size = 0, 0, buf_size
    
    def store(self, obs, act, rew, val, logp):
        """
        Store information for one step interaction
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_v=0):
        """
        Call this function at the end of one simulation episode or at the end 
        of epoch. This function calculates reward-to-go and advantage of 
        the last episode. If the experience does not terminate but 
        the epoch ends, then we use last_v value to bootstrap value of 
        reward-to-go.
        """
        path_slice = slice(self.episode_start, self.ptr)
        
        rew = np.append(self.rew_buf[path_slice], last_v)
        val = np.append(self.val_buf[path_slice], last_v)

        adv_prep = rew[:-1] + self.gamma * val[1:] - val[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(adv_prep, self.gamma * self.lam)
        self.ret_buf[path_slice] = core.discount_cumsum(rew[:-1], self.gamma)

        self.episode_start = self.ptr

    def get(self):
        """
        Only call this function on the end of epoch to get all data, 
        with advantage normalized properly, and to reinitialize for
        next epoch
        """
        assert self.ptr >= self.max_size

        # normalizing adv helps minimizing the variance of the gradient estimation
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        
        self.ptr, self.episode_start = 0, 0

        return {k: torch.as_tensor(v, dtype=torch.float32) 
                for k, v in data.items()}


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, 
        pi_lr=3e-4, vf_lr=1e-3, train_v_iters=80, lam=0.97, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10):
    """
    Inputs:
    env_fn : A function which creates a copy of the environment.
        The environment must satisfy the OpenAI Gym API.
    actor_critic : The constructor method for a PyTorch Module with a 
        ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
        module. The ``step`` method should accept a batch of observations 
        and return:

        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                       | observation.
        ``v``        (batch,)          | Numpy array of value estimates
                                       | for the provided observations.
        ``logp_a``   (batch,)          | Numpy array of log probs for the
                                       | actions in ``a``.
        ===========  ================  ======================================

        The ``act`` method behaves the same as ``step`` but only returns ``a``.

        The ``pi`` module's forward call should accept a batch of 
        observations and optionally a batch of actions, and return:

        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``pi``       N/A               | Torch Distribution object, containing
                                       | a batch of distributions describing
                                       | the policy for the provided observations.
        ``logp_a``   (batch,)          | Optional (only returned if batch of
                                       | actions is given). Tensor containing 
                                       | the log probability, according to 
                                       | the policy, of the provided actions.
                                       | If actions not given, will contain
                                       | ``None``.
        ===========  ================  ======================================

        The ``v`` module's forward call should accept a batch of observations
        and return:

        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``v``        (batch,)          | Tensor containing the value estimates
                                       | for the provided observations. (Critical: 
                                       | make sure to flatten this!)
        ===========  ================  ======================================

    ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
        you provided to VPG.

    seed (int): Seed for random number generators.

    steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
        for the agent and the environment in each epoch.

    epochs (int): Number of epochs of interaction (equivalent to
        number of policy updates) to perform.

    gamma (float): Discount factor. (Always between 0 and 1.)

    pi_lr (float): Learning rate for policy optimizer.

    vf_lr (float): Learning rate for value function optimizer.

    train_v_iters (int): Number of gradient descent steps to take on 
        value function per epoch.

    lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
        close to 1.)

    max_ep_len (int): Maximum length of trajectory / episode / rollout.

    logger_kwargs (dict): Keyword args for EpochLogger.

    save_freq (int): How often (in terms of gap between epochs) to save
        the current policy and value function.

    clip_ratio (float): eps for clip function
    """
    # set up pytorch avoid slowdowns of combo pytorch + mpi
    setup_pytorch_for_mpi()

    # set up logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # set up seed manually
    seed += 1000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # instantiate module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # sync param across processes
    sync_params(ac)

    # count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # set up exprience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # set up function computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        ratio_clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss = -(torch.minimum(ratio * adv, ratio_clip * adv)).mean()

        # useful extra infor
        approx_kl = (logp - logp_old).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss, pi_info

    # set up function computing value function loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()
    
    # set up optimizer for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # set up model saving
    logger.setup_pytorch_saver(ac)

    # one epoch update
    def update():
        data = buf.get()

        pi_old_loss, pi_old_info = compute_loss_pi(data)
        pi_old_loss = pi_old_loss.item()
        v_old_loss = compute_loss_v(data).item()

        # one step update for policy
        pi_optimizer.zero_grad()
        pi_loss, pi_info = compute_loss_pi(data)
        pi_loss.backward()
        mpi_avg_grads(ac.pi)
        pi_optimizer.step()

        # update for v train_v_iters times
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            v_loss = compute_loss_v(data)
            v_loss.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()
        
        kl, ent = pi_info['kl'], pi_old_info['ent']
        logger.store(LossPi=pi_old_loss, LossV=v_old_loss, 
                     KL=kl, Entropy=ent,
                     DeltaLossPi=pi_loss.item() - pi_old_loss,
                     DeltaLossV=v_loss.item() - v_old_loss)
        
    # prepare environment for interaction
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop for PPO
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp_a = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp_a)
            logger.store(VVals=v)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

                if timeout or epoch_ended:
                    # bootstrap if trajectory doesnt end properly
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0

                buf.finish_path(v)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, ep_ret, ep_len = env.reset(), 0, 0

        # save model
        if epoch % save_freq == 0 or epoch == epochs - 1:
            logger.save_state({'env': env}, None)

        # perform PPO update
        update()

        # log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
