import time

import gym
from gym import spaces
import numpy as np
from tensorflow.keras.utils import to_categorical

from .aegis_node import AegisNode

#wraps an AegisNode
#TODO: make extend AegisNode
class AegisEnv(gym.Env):
  def __init__(self, port, obs_url, reward_url, obs_shape, action_shape, discrete=False, n_steps=None):
    self.input_shape = obs_shape
    self.output_shape = action_shape
    self.node = InternalEnvNode(port, obs_url, reward_url)

    #TODO: hardcoded low/highs
    #TODO: handle discrete obs spaces?
    self.observation_space = spaces.Box(shape=[obs_shape], low=-np.Inf, high=np.Inf)
    self.action_space = spaces.Discrete(action_shape) if discrete else spaces.Box(shape=[action_shape], low=-np.Inf, high=np.Inf)
    self.discrete = discrete
    self.n_steps = n_steps
    self.step_count = 0

    self.last_time = time.time()

  def get_observation(self):
    return self.node.get_input("observation", shape=self.observation_space.shape)

  def get_reward(self):
    return self.node.get_input("reward", shape=())

  def step(self, action):
    #calculate niceness time as the time outside of step (RL agent decision time)
    dt = time.time() - self.last_time

    #aegis expects discrete actions to be represented by one-hot (for now)
    if self.discrete:
      action = to_categorical(action, self.action_space.n)
    #set action output for other nodes to pick up
    self.node.set_output(action, "action")

    r = self.get_reward()

    #grab observation
    obs = self.get_observation()

    self.step_count += 1
    done = self.n_steps != None and (self.step_count >= self.n_steps)

    #sleep time equal to update time * niceness + delay
    time.sleep(dt * self.node.niceness)
    time.sleep(self.node.delay)

    self.last_time = time.time()

    return obs, r, done, {}

  def reset(self):
    self.step_count = 0
    self.last_time = time.time()
    return self.get_observation()

#not to be confused with a regular EnvNode
class InternalEnvNode(AegisNode):
  def __init__(self, port, obs_url, reward_url):
    inputs = {
      "observation": obs_url,
      "reward", reward_url,
    }
    outputs = ["action"]
    super().__init__(port, inputs=inputs, outputs=outputs)
