from collections import defaultdict
import threading
import os
import math
import time

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from .aegis_node import AegisNode

import numbers

#TODO: support tensorboard (means/deviations)
#TODO: saving/loading of channels mean/deviation and other parameters?
#TODO: live adjustment of decay rate, scale, clip?
class RewardProxy(AegisNode):
  def __init__(self, port node_urls=[], channels=None, decay_rates=1e-3, clips=3, scales=1, niceness=1, cors=True):
    super().__init__(port, inputs=node_urls, outputs=channels, niceness=niceness, cors=cors)
    self.channels = self.outputs #sanity alias
    self.rewards = {k: 0 for k in self.channels}
    self.means = {k: 0 for k in self.channels}
    self.deviations = {k: 1 for k in self.channels}
    self.niceness = niceness

    #TODO: hardcoded
    self.epsilon = 1e-7

    #allow decay_rates clips, and scales to be single numbers or lists TODO: check len(decay_rates)==len(self.channels)
    self.decay_rates = {k: decay_rates for k in self.channels} if isinstance(decay_rates, numbers.Number) else dict(zip(self.channels, decay_rates))
    self.clips = {k: clips for k in self.channels} if isinstance(clips, numbers.Number) else dict(zip(self.channels, clips))
    self.scales = {k: scales for k in self.channels} if isinstance(scales, numbers.Number) else dict(zip(self.channels, scales))

    #convert half life to decay rate
    self.decay_rates = {k: ln(2) / x if x > 1 else x for k, x in self.decay_rates.items()}

  def update(self):
    sum_reward = 0
    #for each reward
    for channel in self.channels:
      r = self.get_reward(channel)
      mean = self.means[channel]
      deviation = self.deviations[channel]
      decay_rate = self.decay_rates[channel]
      clip = self.clips[channel]
      #apply normalization
      d_reward = r - mean
      normalized_reward = d_reward / (deviation + self.epsilon)
      #update mean/deviation
      self.means[channel] += d_reward * decay_rate
      self.deviations[channel] += (abs(d_reward) - deviation) * decay_rate
      #clip and scale reward
      if clip is not None:
        normalized_reward = np.clip(normalized_reward, -clip, clip)
      normalized_reward *= self.scales[channel]

      sum_reward += normalized_reward

    #distribute rewards
    for channel in self.inputs.keys():
      self.give_reward(sum_reward, channel)
