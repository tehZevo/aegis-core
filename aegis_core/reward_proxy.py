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

class RewardProxy(RequestEngine):
  def __init__(self, node_urls=[], channels=None, niceness=1, port=8181, autostart=True):
    super().__init__(input_urls=node_urls)
    self.channels = [""] if channels is None else channels
    self.rewards = defaultdict(float)
    self.niceness = niceness

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
    #for now, just accumulate
    reward = 0
    #TODO: moving average + scaling
    for name, r in self.rewards.items():
      reward += r
      self.rewards[name] = 0

    #distribute reward, discard tensors.. who needs them anyway?
    self.get_inputs(reward);

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
