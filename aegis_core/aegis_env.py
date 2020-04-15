import time

import gym
from gym import spaces
import numpy as np
from tensorflow.keras.utils import to_categorical

from .aegis_node import AegisNode

#wraps an AegisNode
#TODO: make extend AegisNode
class AegisEnv(gym.Env):
  def __init__(self, node, obs_shape, action_shape, discrete=False,
      n_steps=None, reward_propagation=0):
    self.input_shape = obs_shape
    self.output_shape = action_shape
    self.node = node

    #TODO: hardcoded low/highs
    self.observation_space = spaces.Box(shape=[obs_shape], low=-np.Inf, high=np.Inf)
    self.action_space = spaces.Discrete(action_shape) if discrete else spaces.Box(shape=[action_shape], low=-np.Inf, high=np.Inf)
    self.discrete = discrete
    self.n_steps = n_steps
    self.step_count = 0
    self.reward_propagation = reward_propagation

    self.last_time = time.time()

  def step(self, action):
    #calculate niceness time as the time outside of step (RL agent decision time)
    dt = time.time() - self.last_time

    #aegis expects discrete actions to be represented by one-hot (for now)
    if self.discrete:
      action = to_categorical(action, self.action_space.n)
    #set state for other nodes to pick up
    self.node.set_state(action)

    #get inputs, send rewards, flip received reward buffer
    #see AegisNode.internal_update()
    self.node.pre_update()
    #TODO: callback stuff goes here
    self.node.post_update()

    r = self.node.get_reward()
    self.node.give_reward(r * self.reward_propagation)

    #grab observation
    obs = self.node.get_input()

    self.step_count += 1
    done = self.n_steps != None and (self.step_count >= self.n_steps)

    #sleep time equal to update time * niceness
    if self.niceness >= 0:
      time.sleep(dt * self.niceness)
    else:
      time.sleep(-self.niceness)

    self.last_time = time.time()

    return obs, r, done, {}

  def reset(self):
    self.step_count = 0
    self.last_time = time.time()
    return self.node.get_input()
