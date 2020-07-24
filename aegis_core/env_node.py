import numpy as np
from tensorflow.keras.utils import to_categorical
import warnings

from .aegis_node import AegisNode

#TODO: make done_reward a kwarg
#TODO: better yet, implement done_reward as a gym wrapper instead
class EnvNode(AegisNode):
  def __init__(self, env, port, action_url, done_reward=0,
      action_repeat=1, render=False, obs_scale=None):
    """Note: for discrete environments, uses argmax for action selection"""
    inputs = { "action": action_url }
    outputs = ["observation", "reward", "done", "action", "episode_reward"]

    super().__init__(port, inputs=inputs, outputs=outputs)

    self.env = env
    self.done_reward = done_reward
    self.action_repeat = action_repeat
    self.render = render

    self.action_warn = True

    self.obs_scale = obs_scale

    self.is_discrete = self.env.action_space.shape == ()
    self.episode_reward = 0

    # self.output_shape = self.env.observation_space.shape
    # if self.is_discrete:
    #   self.input_shape = [self.env.action_space.n]
    # else:
    #   self.input_shape = self.env.action_space.shape

    self.env.reset()

  def get_dummy_action(self):
    if self.is_discrete:
      return to_categorical(np.random.choice(self.env.action_space.n))

    return np.zeros(self.env.action_space.shape)

  def get_action(self):
    action = self.get_input("action")
    if action is None:
      if self.action_warn:
        warnings.warn("Action is none")
        self.action_warn = False
    else:
      self.action_warn = True
      action = self.get_dummy_action()

    #housekeeping
    action = action.astype("float32")

    return action

  def update(self):
    cb_data = {}
    #get input
    action = self.get_action()

    og_action = action

    #step env
    if self.is_discrete:
      action = np.argmax(action)

    r = 0
    for i in range(self.action_repeat):
      state, reward, done, info = self.env.step(action)
      r += reward
      if done:
        break

    if done:
      r += self.done_reward

    self.episode_reward += r

    if done:
      state = self.env.reset()

    #temporary sanity for atari ram / image observations
    if self.obs_scale is not None:
      state = self.obs_scale(state)

    #set outputs
    self.set_output(r, "reward")
    self.set_output(state, "observation")

    #sel logging outputs
    self.set_output(1 if done else 0, "done") #TODO: can we support ints/floats here?
    self.set_output(og_action, "action")
    self.set_output(self.episode_reward, "episode_reward")

    if self.render:
      self.env.render()

    if done:
      self.episode_reward = 0

if __name__ == "__main__":
  import logging
  import gym
  from .utils import start_nodes
  from .random_node import RandomNode

  #log = logging.getLogger('werkzeug')
  #log.setLevel(logging.ERROR)

  #random agent
  env = gym.make("LunarLander-v2")
  start_nodes([
    RandomNode(12399, shape=[env.action_space.n]),
    EnvNode(env, 12400, "12399", render=True)
  ])
