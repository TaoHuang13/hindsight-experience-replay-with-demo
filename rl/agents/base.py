import copy
import torch
import numpy as np


class BaseAgent(object):
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def get_samples(self, replay_buffer):
		# Sample replay buffer 
        transitions = replay_buffer.sample()

        # preprocess
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        obs = self.to_torch(inputs_norm)
        next_obs = self.to_torch(inputs_next_norm)
        action = self.to_torch(transitions['actions'])
        reward = self.to_torch(transitions['r'])

        return obs, action, reward, next_obs

    def update_normalizer(self, episode_batch):        
        mb_obs, mb_ag, mb_g, mb_actions, done = episode_batch

        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_sampler.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preproc_inputs(self, o, g):
        '''State normalization'''

        o_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)
 
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return inputs

    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array, dtype=torch.float32).to(self.device)
        return torch.as_tensor(array).to(self.device)

    def save(self, model_dir, step):
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))
        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))

