import numpy as np
from tensorflow.keras.utils import to_categorical

from ml_utils.viz import save_plot

from .engine import RequestEngine

#TODO: action repeat?

class EnvEngine(RequestEngine):
  def __init__(self, env, end_reward, action_url, run_name="",
      viz_interval=100, viz_quantile=0.05):
    super().__init__(input_urls=[action_url])

    self.env = env;
    self.end_reward = end_reward
    self.run_name = run_name

    #TODO: make viz_interval steps instead of episodes
    self.viz_interval = viz_interval
    self.viz_quantile = viz_quantile

    self.last_reward = 0
    self.is_discrete = self.env.action_space.shape == ()
    self.input_size = self.env.observation_space.shape [0] #TODO: handle multidim

    if self.is_discrete:
      self.output_size = self.env.action_space.n
    else:
      self.output_size = self.env.action_space.shape[0] #TODO: handle multidim

    self.episode_rewards = []
    self.step_rewards = []

    self.episode_actions = []
    self.step_actions = []

    self.env.reset()

  def get_action(self, reward):
    action = self.get_inputs(reward)[0] #only one input, only one action

    if action is None:
      if self.is_discrete:
        #TODO: clean up.. for now, just return random action if failed to get action
        action = to_categorical(np.random.randint(self.env.action_space.n))
      else:
        action = np.zeros(self.env.action_space.shape)

    return action

  def update(self, reward):
    #get input (and give reward :)))) )
    action = self.get_action(self.last_reward);

    #housekeeping
    action = action.astype("float32")

    self.step_actions.append(action)

    #step env
    if self.is_discrete:
      action = np.argmax(action)

    state, last_reward, done, info = self.env.step(action)
    if done:
      last_reward += self.end_reward
    self.last_reward = last_reward
    self.step_rewards.append(last_reward)

    if done:
      state = self.env.reset()
      episode_reward = np.sum(self.step_rewards)
      self.episode_rewards.append(episode_reward)

      episode_action = np.mean(self.step_actions, axis=0)
      self.episode_actions.append(episode_action)
      self.step_actions = []

      print("Ep {}: {}".format(len(self.episode_rewards), episode_reward))
      self.step_rewards = []

      if len(self.episode_rewards) % self.viz_interval == 0:
        save_plot(self.episode_rewards, "{} Episode rewards".format(self.run_name), 0.01, q=self.viz_quantile)
        save_plot(self.episode_actions, "{} Actions".format(self.run_name), 0.01)

    return state
