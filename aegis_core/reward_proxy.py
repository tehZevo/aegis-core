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
  def __init__(self, port, channels, decay_rates=1e-3, clips=3, scales=1):
    """Channels should be a dict of channel_name:reward_url"""
    super().__init__(port, inputs=channels)
    self.channels = self.inputs.keys() #sanity alias
    self.rewards = {k: 0 for k in self.channels}
    self.means = {k: 0 for k in self.channels}
    self.deviations = {k: 1 for k in self.channels}
    self.niceness = niceness

    #TODO: hardcoded
    self.epsilon = 1e-7

    #allow decay_rates clips, and scales to be single numbers or lists TODO: check len(decay_rates)==len(self.channels)
    #TODO: check that all keys are present in decay_rates/clips/scales (as in channels)
    self.decay_rates = {k: decay_rates for k in self.channels} if isinstance(decay_rates, numbers.Number) else decay_rates
    self.clips = {k: clips for k in self.channels} if isinstance(clips, numbers.Number) else clips
    self.scales = {k: scales for k in self.channels} if isinstance(scales, numbers.Number) else scales

    #convert half life to decay rate (values > 1)
    self.decay_rates = {k: ln(2) / x if x > 1 else x for k, x in self.decay_rates.items()}

  def update(self):
    sum_reward = 0
    #for each reward
    for channel in self.channels:
      r = self.get_input(channel, ()) #default to 0 if error
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

    #set reward value
    self.set_output(sum_reward)

if __name__ == "__main__":
  import logging

  log = logging.getLogger('werkzeug')
  log.setLevel(logging.ERROR)

  RewardProxy(12400, []).start()
