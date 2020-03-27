import numpy as np
from tensorflow.keras.utils import to_categorical

from .engine import RequestEngine, sanitize

#TODO: make done_reward a kwarg
class EnvEngine(RequestEngine):
  def __init__(self, env, done_reward, action_url, run_name="",
      reward_proxy=None, action_repeat=1, render=False, obs_scale=None,
      callbacks=[]):
    super().__init__(input_urls=[action_url])

    self.env = env
    self.done_reward = done_reward
    self.run_name = run_name
    self.reward_proxy = None if reward_proxy is None else sanitize(reward_proxy)
    self.action_repeat = action_repeat
    self.render = render

    self.obs_scale = obs_scale
    self.callbacks = callbacks

    self.last_reward = 0
    self.is_discrete = self.env.action_space.shape == ()
    self.output_shape = self.env.observation_space.shape

    if self.is_discrete:
      self.input_shape = [self.env.action_space.n]
    else:
      self.input_shape = self.env.action_space.shape

    #TODO: flag to store episode stats or not
    self.step_rewards = []
    self.step_actions = []

    self.env.reset()

  def get_action(self, reward):
    action = self.get_inputs(reward)[0] #only one input, only one action

    if action is None:
      if self.is_discrete:
        #TODO: clean up.. for now, just return random action if failed to get action
        action = to_categorical(np.random.randint(self.env.action_space.n), self.env.action_space.n)
      else:
        action = np.zeros(self.env.action_space.shape)

    return action

  def update(self, reward):
    cb_data = {}

    #if we have a reward proxy, send the reward there instead of the upstream node
    if self.reward_proxy is not None:
      self.get_single_input(self.reward_proxy, self.last_reward)
      self.last_reward = 0; #nuke reward so upstream nodes get 0

    #get input (and give reward :)))) )
    action = self.get_action(self.last_reward);

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

    self.last_reward = r
    self.step_rewards.append(r)

    if done:
      state = self.env.reset()

    cb_data["reward"] = r
    cb_data["done"] = done
    cb_data["step_rewards"] = self.step_rewards
    cb_data["step_actions"] = self.step_actions
    cb_data["state"] = state
    cb_data["action"] = og_action
    cb_data["engine"] = self

    for cb in self.callbacks:
      cb(cb_data)

    if done:
      self.step_rewards = []
      self.step_actions = []

    if self.render:
      self.env.render()

    #temporary sanity for atari ram
    if self.obs_scale is not None:
      state = self.obs_scale(state)

    return state
