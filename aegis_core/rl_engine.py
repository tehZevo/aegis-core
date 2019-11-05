import numpy as np
import tensorflow as tf

from .engine import RequestEngine

class RLEngine(RequestEngine):
  def __init__(self, agent, input_urls=[], train=True, reward_propagation=0,
      reward_transform=lambda x: x, callbacks=[]):
    super().__init__(input_urls=input_urls)

    self.agent = agent
    self.do_train = train
    self.reward_transform = reward_transform
    self.reward_propagation = reward_propagation
    self.callbacks = callbacks

    self.input_shape = tuple(self.agent.model.input_shape[1:])
    self.output_shape = tuple(self.agent.model.output_shape[1:])

  def update(self, reward):
    transformed_reward = self.reward_transform(reward)

    if self.do_train:
      self.agent.train(transformed_reward)

    prop_reward = transformed_reward #TODO: use untransformed reward?

    #TODO: other reward propagation schemes?
    input_states = self.get_inputs(prop_reward * self.reward_propagation / len(self.input_urls))
    #fix Nones
    input_states = [np.zeros(self.input_shape) if x is None else x for x in input_states]
    input_state = np.mean(input_states, axis=0) #TODO: this will fail for discrete where a state is missing (set to 0s...) actually maybe not
    output_state = self.agent.get_action(input_state)

    cb_values = {}
    cb_values["agent"] = self.agent

    for cb in self.callbacks:
      cb(cb_values)

    return output_state
