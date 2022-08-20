import hydra
from pathlib import Path
import rl.utils.utils as utils
import torch
from rl.agents.factory import make_agent
from rl.modules.logger import Logger
from rl.modules.replay_buffer import HerReplayBuffer
from rl.utils.video_recorder import VideoRecorder
import gym


class Experiment:
    def __init__(self, cfg):
        self.work_dir = Path(cfg.cwd)

        # TODO: use wandb logger
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup(cfg)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    def setup(self, cfg):
        # create logger
        self.logger = Logger(self.work_dir)

        # create env
        self.train_env = gym.make(self.cfg.task_name)
        self.eval_env = gym.make(self.cfg.task_name)

        env_params = utils.get_env_params(self.train_env)
        sampler = utils.get_sampler(self.train_env, self.cfg.agent.sampler)

        # create agent
        self.agent = make_agent(
            env_params=env_params,
            sampler=sampler,
            cfg=cfg.agent
        )

        # create buffer
        self.buffer = HerReplayBuffer(
            buffer_size=cfg.replay_buffer_capacity,
            env_params=env_params,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
        )
        self.demo_buffer = HerReplayBuffer(
            buffer_size=cfg.replay_buffer_capacity,
            env_params=env_params,
            batch_size=self.cfg.batch_size,
            sampler=sampler
        )

        # initialized demo buffer
        utils.init_buffer_norm(cfg, self.demo_buffer, self.agent)

        # create video recorder
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_steps)
        seed_until_step = utils.Until(self.cfg.num_seed_steps)
        log_every_step = utils.Every(self.cfg.log_frequency)
        eval_every_step = utils.Every(self.cfg.eval_frequency)

        episode_step, episode_reward, done = 0, 0, False
        episode_cache = utils.ReplayCache(self.train_env._max_episode_steps)
        metrics = None

        obs = self.train_env.reset()
        episode_cache.store_obs(obs)

        while train_until_step(self.global_step):
            if done:
                episode = episode_cache.pop()
                self.buffer.store_episode(episode)
                self.agent.update_normalizer(episode)
                self._global_episode += 1

                # wait until all the metrics schema is populated
                if metrics is not None and log_every_step(self.global_step):
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                        log('fps', episode_step / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_step)
                        log('episode', self.global_episode)
                        log('step', self.global_step)

                # reset env
                obs = self.train_env.reset()
                episode_cache.store_obs(obs)
                episode_step = 0
                episode_reward = 0

                # try to update the agent
                if not seed_until_step(self.global_step):
                    metrics = self.agent.update(self.buffer, self.demo_buffer)
                    #metrics = self.agent.update_bc(self.demo_buffer)
                    self.logger.log_metrics(metrics, self.global_step, ty='train')

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_step)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.get_action(obs, self.global_step, noise=True)

            # take env step
            obs, reward, done, _ = self.train_env.step(action)
            episode_cache.store_transition(obs, action, done)
            episode_reward += reward
            episode_step += 1
            self._global_step += 1

    def eval(self):
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs = self.eval_env.reset()
            done = False
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.get_action(obs, self.global_step, noise=False)
                obs, reward, done, info = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1

            total_sr += int(info['is_success'])
            episode += 1
            self.video_recorder.save(f'{self.global_step}.mp4')

        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step / episode)
            log('episode', self.global_episode)
            log('success_rate', total_sr / episode)
            log('step', self.global_step)

    def save_checkpoints(self):
        pass

    def load_checkpoints(self):
        pass
    

@hydra.main(version_base=None, config_path="./configs", config_name="HerDemo")
def main(cfg):
    exp = Experiment(cfg)
    exp.train()

if __name__ == "__main__":
    main()