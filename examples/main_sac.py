import oatomobile
import oatomobile.envs
import oatomobile.baselines.rulebased
from action_dict_to_array import ActionDictToArray
from jaxrl.wrappers import TakeKey
from tqdm import tqdm


import gym
import os
import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import DrQLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
import jaxrl.wrappers
from jaxrl.wrappers import VideoRecorder

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', 1,
    'Action repeat.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'drq_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


class VelocityReward(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = np.linalg.norm(info['ignored_observations']['velocity'])
        return observation, reward, done, info


class AllGasNoBrakes(gym.ActionWrapper):
    def action(self, action):
        new_action = action.copy()
        new_action[0] = -1.0
        new_action[2] = 1.0
        return new_action


def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    def make_pixel_env(seed):
        env = oatomobile.envs.CARLAEnv(town="Town01", sensors=['front_camera_rgb', 'velocity'])
        env = ActionDictToArray(env)
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
        env = AllGasNoBrakes(env)
        env = TakeKey(env, 'front_camera_rgb')
        env = VelocityReward(env)
        env = jaxrl.wrappers.FrameStack(env, 3)
        env = gym.wrappers.TimeLimit(env, 200)
        env = jaxrl.wrappers.EpisodeMonitor(env)
        return env

    env = make_pixel_env(FLAGS.seed)
    eval_env = make_pixel_env(FLAGS.seed + 42)
    video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    eval_env = VideoRecorder(eval_env, save_folder=video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    agent = DrQLearner(FLAGS.seed,
                       env.observation_space.sample()[np.newaxis],
                       env.action_space.sample()[np.newaxis], **kwargs)

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, (FLAGS.max_steps + 1) // FLAGS.action_repeat),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
  
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

if __name__ == '__main__':
    app.run(main)