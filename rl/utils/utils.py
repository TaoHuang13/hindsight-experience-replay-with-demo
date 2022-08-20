import torch
import numpy as np
import os
import random
from rl.modules.replay_buffer import HER_sampler
import time

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_env_params(env):
    obs = env.reset()
    get_env_params = dict(
        obs=obs['observation'].shape[0],
        goal=obs['desired_goal'].shape[0],
        act=env.action_space.shape[0],
        act_rand_sampler=env.action_space.sample,
        max_timesteps=env._max_episode_steps,
        max_action=env.action_space.high[0]
    )
    return get_env_params

def get_sampler(env, cfg):
    '''Sampler of replay buffer'''
    if cfg.type == 'her':
        sampler = HER_sampler(
            replay_strategy=cfg.strategy,
            replay_k=cfg.k,
            reward_func=env.compute_reward,
        )
    else:
        raise NotImplementedError

    return sampler


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class ReplayCache:
    '''Cache of episode during training'''

    def __init__(self, T):
        self.T = T
        self.reset()

    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones = [], [], [], [], []

    def store_transition(self, obs, action, done):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)

    def store_obs(self, obs):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)

        self.reset()
        return (obs, ag, g, actions, dones)


def init_buffer_norm(cfg, buffer, agent):
    '''Load demonstrations into buffer and initilaize normalizer'''

    demo_path = os.path.join(os.getcwd(),'surrol/data/demo')
    file_name = "data_"
    file_name += cfg.task_name
    file_name += "_" + 'random'
    file_name += "_" + str(cfg.num_demo)
    file_name += ".npz"

    demo_path = os.path.join(demo_path, file_name)
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs, demo_info = demo['obs'], demo['acs'], demo['info']

    episode_cache = ReplayCache(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])

        for i in range(buffer.T):
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i==(buffer.T-1)
            )

        episode = episode_cache.pop()
        buffer.store_episode(episode)
        agent.update_normalizer(episode)
