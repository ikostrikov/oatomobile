import os
import random

import carla
import gym
import jaxrl.wrappers
import numpy as np
import oatomobile
import oatomobile.baselines.rulebased
import oatomobile.envs
import tqdm
from absl import app, flags
from agents.navigation.controller import PIDLongitudinalController
from jaxrl.agents import DrQLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.wrappers import TakeKey, VideoRecorder
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from tqdm import tqdm

from action_dict_to_array import ActionDictToArray

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
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
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


class AccelerationControl(gym.ActionWrapper):
    def __init__(self, env, max_throttle=0.75, max_brake=0.3):
        super().__init__(env)
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        spaces = self.env.action_space.spaces.copy()
        spaces.pop('brake')
        spaces.pop('throttle')
        self.action_space = gym.spaces.Dict({
            **spaces, 'acceleration':
            gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
        })

    def action(self, action):
        new_action = action.copy()
        acceleration = new_action.pop('acceleration')
        if acceleration >= 0.0:
            new_action['throttle'] = min(acceleration, self.max_throttle)
            new_action['brake'] = 0.0
        else:
            new_action['throttle'] = 0.0
            new_action['brake'] = min(abs(acceleration), self.max_brake)

        return new_action


class VelocityControl(gym.ActionWrapper):
    def __init__(self, env, max_velocity=200.0):
        super().__init__(env)

        spaces = self.env.action_space.spaces.copy()
        spaces.pop('acceleration')
        self.action_space = gym.spaces.Dict({
            **spaces, 'target_velocity':
            gym.spaces.Box(low=0.0,
                           high=max_velocity,
                           shape=(),
                           dtype=np.float32)
        })

        # Unwrap to reach the sim.
        sim = self.env
        while True:
            if hasattr(sim, '_sim'):
                sim = sim._sim
                break
            else:
                sim = sim.env
        self.sim = sim
        self._reset_controller()

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._reset_controller()
        return obs

    def _reset_controller(self):
        self._dt = 1.0 / 20.0

        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt
        }

        self._lon_controller = PIDLongitudinalController(
            self.sim.hero, **args_longitudinal_dict)

    def action(self, action):
        new_action = action.copy()
        target_velocity = new_action.pop('target_velocity')
        acceleration = self._lon_controller.run_step(target_velocity)
        return {**new_action, 'acceleration': acceleration}


def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    def make_pixel_env(seed):
        env = oatomobile.envs.CARLAEnv(
            town="Town01", sensors=['front_camera_rgb', 'velocity'])
        env = AccelerationControl(env)
        # env = VelocityControl(env)
        env = ActionDictToArray(env)
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
        env = TakeKey(env, 'front_camera_rgb')
        env = VelocityReward(env)
        env = jaxrl.wrappers.FrameStack(env, 3)
        env = gym.wrappers.TimeLimit(env, 200)
        env = jaxrl.wrappers.EpisodeMonitor(env)
        return env

    video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    env = make_pixel_env(FLAGS.seed)
    env = VideoRecorder(env, save_folder=video_eval_folder)
    # eval_env = make_pixel_env(FLAGS.seed + 42)
    # eval_env = VideoRecorder(eval_env, save_folder=video_eval_folder)

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
    for i in tqdm(range(1, (FLAGS.max_steps + 1) // FLAGS.action_repeat),
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
        """
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
        """


if __name__ == '__main__':
    app.run(main)
