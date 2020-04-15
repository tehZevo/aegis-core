import numpy as np
from tensorflow.keras.utils import to_categorical
import warnings

from .aegis_node import AegisNode

#TODO: make done_reward a kwarg
#TODO: better yet, implement done_reward as a gym wrapper instead
class EnvNode(AegisNode):
  def __init__(self, env, port, action_urls, run_name="", done_reward=0,
      reward_proxy=None, action_repeat=1, render=False, obs_scale=None,
      callbacks=None, cors=True, niceness=1):
    """Note: for discrete environments, uses argmax for action selection"""
    inputs = { "action": action_urls }
    if reward_proxy is not None:
      inputs["reward_proxy"] = reward_proxy

    super().__init__(port, inputs=inputs, niceness=niceness, cors=cors, callbacks=callbacks)

    self.env = env
    self.done_reward = done_reward
    self.run_name = run_name
    self.action_repeat = action_repeat
    self.render = render
    self.use_reward_proxy = reward_proxy is not None

    self.action_warn = True

    self.obs_scale = obs_scale

    self.is_discrete = self.env.action_space.shape == ()

    # self.output_shape = self.env.observation_space.shape
    # if self.is_discrete:
    #   self.input_shape = [self.env.action_space.n]
    # else:
    #   self.input_shape = self.env.action_space.shape

    #TODO: flag to store episode stats or not
    self.step_rewards = []
    self.step_actions = []

    self.env.reset()

  def get_dummy_action(self):
    if self.is_discrete:
      return to_categorical(np.random.choice(self.env.action_space.n))

    return np.zeros(self.env.action_space.shape)

  def update(self):
    cb_data = {}
    #get input
    action = self.get_input("action")
    if len(action) == 0 or action[0] is None:
      if self.action_warn:
        warnings.warn("Action is zero-length or none")
        self.action_warn = False
      action = self.get_dummy_action()
    else:
      self.action_warn = True

    #housekeeping
    action = action.astype("float32")
    og_action = action
    self.step_actions.append(action)

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

    #if we have a reward proxy, send the reward there instead of the upstream node
    if self.use_reward_proxy:
      #TODO: define reward?
      self.give_reward(r, "reward_proxy")
    else:
      self.give_reward(r, "action")

    self.step_rewards.append(r)

    if done:
      state = self.env.reset()

    cb_data["reward"] = r
    cb_data["done"] = done
    cb_data["step_rewards"] = self.step_rewards
    cb_data["step_actions"] = self.step_actions
    cb_data["state"] = state
    cb_data["action"] = og_action
    cb_data["node"] = self

    self.set_callback_data(cb_data)

    if done:
      self.step_rewards = []
      self.step_actions = []

    if self.render:
      self.env.render()

    #temporary sanity for atari ram / image observations
    if self.obs_scale is not None:
      state = self.obs_scale(state)

    self.set_state(state)

if __name__ == "__main__":
  import logging
  import gym

  #log = logging.getLogger('werkzeug')
  #log.setLevel(logging.ERROR)

  env = gym.make("LunarLander-v2")
  EnvNode(env, 12400, []).start()
