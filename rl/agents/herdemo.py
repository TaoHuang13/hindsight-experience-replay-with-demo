import torch
import torch.nn.functional as F
import copy
import numpy as np
from rl.utils.normalizer import Normalizer
from rl.modules.policies import DeterministicPolicy, Critic
from rl.agents.base import BaseAgent


class HerDemo(BaseAgent):
    def __init__(
        self,
        env_param,
        sampler,
        agent_cfg,
    ):
        self.discount = agent_cfg.discount
        self.reward_scale = agent_cfg.reward_scale
        self.log_interval = agent_cfg.log_interval
        self.update_epoch = agent_cfg.update_epoch
        self.random_steps = agent_cfg.num_seed_steps
        self.her_sampler = sampler    # same as which in buffer
        self.device = agent_cfg.device

        self.random_eps = agent_cfg.random_eps
        self.noise_eps = agent_cfg.noise_eps
        self.aux_weight = agent_cfg.aux_weight
        self.action_l2 = agent_cfg.action_l2
        self.p_dist = agent_cfg.p_dist
        self.soft_target_tau = agent_cfg.soft_target_tau

        self.clip_obs = agent_cfg.clip_obs
        self.norm_clip = agent_cfg.norm_clip
        self.norm_eps = agent_cfg.norm_eps

        self.dima = env_param['act']
        self.dimo, self.dimg = env_param['obs'], env_param['goal']

        self.max_action = env_param['max_action']
        self.act_sampler = env_param['act_rand_sampler']

        # normarlizer
        self.o_norm = Normalizer(
            size=self.dimo, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )
        self.g_norm = Normalizer(
            size=self.dimg, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )

        # build policy
        self.actor = DeterministicPolicy(
            self.dimo, self.dimg, self.dima, agent_cfg.hidden_dim).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)

        self.critic = Critic(
            self.dimo, self.dimg, self.dima, agent_cfg.hidden_dim).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )

        self.train()
        self.critic_target.train()

    def get_action(self, state, global_step, noise=False):
        # random action at initial stage
        if global_step < self.random_steps:
            return self.act_sampler()

        with torch.no_grad():
            o, g = state['observation'], state['desired_goal']
            input_tensor = self._preproc_inputs(o, g)
            action = self.actor(input_tensor).cpu().data.numpy().flatten()

            # Gaussian noise
            if noise:
                action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
                    -self.max_action, self.max_action)

        return action
    
    def update_critic(self, obs, action, reward, next_obs):
        metrics = dict()

        with torch.no_grad():
            action_out = self.actor_target(next_obs)
            target_V = self.critic_target(next_obs, action_out)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

            clip_return = 1 / (1 - self.discount)
            target_Q = torch.clamp(target_Q, -clip_return, 0).detach()

        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        # Optimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()      
        
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = Q.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        return metrics

    def update_actor(self, obs, action, is_demo=False):
        metrics = dict()

        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)

        # Refer to https://arxiv.org/pdf/1709.10089.pdf
        if is_demo:
            bc_loss = self.norm_dist(action_out, action)

            # Q-filter
            with torch.no_grad():
                q_filter = self.critic_target(obs, action) > self.critic_target(obs, action_out)
            bc_loss = q_filter * bc_loss
            
            actor_loss = -(Q_out + self.aux_weight * bc_loss).mean()
        else:
            actor_loss = -(Q_out).mean()

        # Optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_epoch):
            # Sample from replay buffer 
            obs, action, reward, next_obs = self.get_samples(replay_buffer)
    
            # Update critic and actor
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor(obs, action))

            # # Sample from demo buffer 
            obs, action, reward, next_obs = self.get_samples(demo_buffer)

            # Update critic
            self.update_critic(obs, action, reward, next_obs)
            self.update_actor(obs, action, is_demo=True)

        # Update target critic and actor
        self.update_target()

        return metrics

    def update_target(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

    def norm_dist(self, a1, a2):
        #return - (a1 - a2).pow(2).sum(dim=1,keepdim=True) / self.action_dim
        self.p_dist = np.inf if self.p_dist == -1 else self.p_dist
        return - torch.norm(a1 - a2, p=self.p_dist, dim=1, keepdim=True).pow(2) / self.dima 