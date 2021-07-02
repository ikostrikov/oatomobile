import gym
from jaxrl.wrappers.dmc_env import _spec_to_box, _flatten, _unflatten
import numpy as np

class ActionDictToArray(gym.ActionWrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Dict), (
            "expected Dict action space, got {}".format(type(env.action_space)))
        super(ActionDictToArray, self).__init__(env)

        self.old_action_space = env.action_space

        self.action_space = _spec_to_box(self.action_space.spaces.values())

    def action(self, action):
        return _unflatten(self.old_action_space, action)
