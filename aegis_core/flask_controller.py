import numpy as np
import threading
import os
import math
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from .controller import Controller

class ControllerResource(Resource):
  def __init__(self, controller, channel):
    self.controller = controller
    self.channel = channel

  def post(self):
    reward = request.get_json(force=True)
    if reward is None:
      print("reward was none!")
      os._exit(1)
    if math.isnan(reward):
      print("reward was nan!")
      os._exit(1)

    self.controller.rewards[self.channel] += reward
    data = self.controller.state
    if data is None:
      data = np.zeros([self.controller.engine.output_size]) #TODO: 1d
    #print("DATAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", data)
    if np.isnan(data).any():
      print("OH NO, NANS IN OUTPUT! {}".format(data))
      os._exit(1)
    data = jsonify(data.tolist())

    return data

class FlaskController(Controller):
  def __init__(self, engine, niceness=1, port=8181, channels=None, autostart=True):
    super().__init__(engine, niceness=niceness)

    self.channels = [""] if channels is None else channels
    self.rewards = defaultdict(float)

    self.start_server(port)

    if autostart:
      self.loop()

  def pre_step(self):
    #accumulate rewards from channels
    for name, reward in self.rewards.items():
      self.reward += reward
      self.rewards[name] = 0

  def start_server(self, port):
    flask_app = Flask(__name__)
    api = Api(flask_app)

    for channel in self.channels:
      api.add_resource(ControllerResource, "/{}".format(channel),
        resource_class_kwargs={"controller": self, "channel": channel})

    self.flask_app = flask_app

    def dedotated_wam():
      self.flask_app.run(debug=False, threaded=True, port=port)

    self.app_thread = threading.Thread(target=dedotated_wam)
    self.app_thread.daemon = True
    self.app_thread.start()
