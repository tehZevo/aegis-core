from collections import defaultdict
import threading
import os
import math
import time

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from .engine import RequestEngine

class RewardResource(Resource):
  def __init__(self, proxy, channel):
    self.proxy = proxy
    self.channel = channel

  def post(self):
    reward = request.get_json(force=True)
    if reward is None:
      print("reward was none!")
      os._exit(1)
    if math.isnan(reward):
      print("reward was nan!")
      os._exit(1)

    #add reward to channel
    self.proxy.rewards[self.channel] += reward
    data = np.zeros([0]) #for sanity
    data = jsonify(data.tolist())

    return data

import numbers

#TODO: expose means/deviations for inspection via http or something
#TODO: saving/loading of channels mean/deviation and other parameters?
#TODO: live adjustment of decay rate, scale, clip?
class RewardProxy(RequestEngine):
  def __init__(self, node_urls=[], channels=None, decay_rates=1e-3, clips=3, scales=1, niceness=1, port=8181, autostart=True):
    super().__init__(input_urls=node_urls)
    self.channels = [""] if channels is None else channels
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

    self.start_server(port)

    if autostart:
      self.loop()

  def loop(self):
    while True:
      starttime = time.time()
      self.update()
      #sleep time equal to update time * niceness
      dt = time.time() - starttime
      if self.niceness >= 0:
        time.sleep(dt * self.niceness)
      else:
        time.sleep(-self.niceness)

  def update(self):
    sum_reward = 0
    #for each reward
    for name, r in self.rewards.items():
      mean = self.means[name]
      deviation = self.deviations[name]
      decay_rate = self.decay_rates[name]
      clip = self.clips[name]
      #apply normalization
      d_reward = r - mean
      normalized_reward = d_reward / (deviation + self.epsilon)
      #update mean/deviation
      self.means[name] += d_reward * decay_rate
      self.deviations[name] += (deviation - abs(d_reward)) * decay_rate
      #clip and scale reward
      if clip is not None:
        normalized_reward = np.clip(normalized_reward, -clip, clip)
      normalized_reward *= self.scales[name]

      sum_reward += normalized_reward
      self.rewards[name] = 0

    #distribute reward, discard tensors.. who needs them anyway?
    self.get_inputs(sum_reward)

  #TODO: clean duped code from flask controller...
  def start_server(self, port):
    flask_app = Flask(__name__)
    api = Api(flask_app)

    for channel in self.channels:
      api.add_resource(RewardResource, "/{}".format(channel), endpoint=channel,
        resource_class_kwargs={"proxy": self, "channel": channel})

    self.flask_app = flask_app

    def dedotated_wam():
      self.flask_app.run(debug=False, threaded=True, port=port)

    self.app_thread = threading.Thread(target=dedotated_wam)
    self.app_thread.daemon = True
    self.app_thread.start()
