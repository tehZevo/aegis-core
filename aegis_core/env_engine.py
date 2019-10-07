import numpy as np
from tensorflow.keras.utils import to_categorical

from ml_utils.viz import save_plot

from .engine import RequestEngine, sanitize

#TODO: action repeat?

class EnvEngine(RequestEngine):
  def __init__(self, env, end_reward, action_url, run_name="",
      viz_quantile=0.05, viz_smoothing=0.1, reward_proxy=None,
      action_repeat=1, draw_raw_actions=True, render=False, report_interval=1,
      interval_type="episode", obs_scale=None):
    super().__init__(input_urls=[action_url])

    self.env = env;
    self.end_reward = end_reward
    self.run_name = run_name
    self.reward_proxy = None if reward_proxy is None else sanitize(reward_proxy)
    self.action_repeat = action_repeat
    self.draw_raw_actions = draw_raw_actions
    self.render = render

    self.viz_quantile = viz_quantile
    self.viz_smoothing = viz_smoothing

    self.report_interval = report_interval
    self.interval_type = interval_type
    self.interval_counter = 0

    self.obs_scale = obs_scale

    self.last_reward = 0
    self.is_discrete = self.env.action_space.shape == ()
    self.output_shape = self.env.observation_space.shape

    if self.is_discrete:
      self.input_shape = [self.env.action_space.n]
    else:
      self.input_shape = self.env.action_space.shape

    self.interval_rewards = []
    self.step_rewards = []

    self.interval_actions = []
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
    #if we have a reward proxy, send the reward there instead of the upstream node
    if self.reward_proxy is not None:
      self.get_single_input(self.reward_proxy, self.last_reward)
      self.last_reward = 0; #nuke reward so upstream nodes get 0

    #get input (and give reward :)))) )
    action = self.get_action(self.last_reward);

    #housekeeping
    action = action.astype("float32")

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
      r += self.end_reward

    self.last_reward = r
    self.step_rewards.append(r)

    if done:
      state = self.env.reset()

    #TODO: use callbacks...
    if (self.interval_type == "episode" and done) or self.interval_type == "step":
      self.interval_counter += 1

    if self.interval_counter >= self.report_interval:
      self.interval_counter = 0
      interval_reward = np.sum(self.step_rewards)
      self.interval_rewards.append(interval_reward)

      interval_action = np.mean(self.step_actions, axis=0)
      self.interval_actions.append(interval_action)

      step_or_ep = "Episode" if self.interval_type == "episode" else "Interval"
      print("{} {}: {} ({} steps)".format(step_or_ep, len(self.interval_rewards), interval_reward, len(self.step_rewards)))
      self.step_rewards = []
      self.step_actions = []

      save_plot(self.interval_rewards, "{} {} Rewards".format(self.run_name, step_or_ep),
        self.viz_smoothing, q=self.viz_quantile)
      save_plot(self.interval_actions, "{} {} Actions".format(self.run_name, step_or_ep),
        self.viz_smoothing, draw_raw=self.draw_raw_actions)

    if self.render:
      self.env.render()

    #temporary sanity for atari ram
    if self.obs_scale is not None:
      state = self.obs_scale(state)

    return state
