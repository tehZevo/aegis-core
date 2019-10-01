import numpy as np
import threading
import os
import math

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from .controller import Controller

class ControllerResource(Resource):
  def __init__(self, controller):
    self.controller = controller

  def post(self):
    reward = request.get_json(force=True)
    if reward is None:
      print("reward was none!")
      os._exit(1)
    if math.isnan(reward):
      print("reward was nan!")
      os._exit(1)

    self.controller.reward += reward
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
  def __init__(self, engine, niceness=1, port=8181, autostart=True):
    super().__init__(engine, niceness=niceness)
    self.start_server(port)

    if autostart:
      self.loop()

  def start_server(self, port):
    flask_app = Flask(__name__)
    api = Api(flask_app)

    api.add_resource(ControllerResource, "/", resource_class_kwargs={"controller": self})

    self.flask_app = flask_app

    def dedotated_wam():
      self.flask_app.run(debug=False, threaded=True, port=port)

    self.app_thread = threading.Thread(target=dedotated_wam)
    self.app_thread.daemon = True
    self.app_thread.start()
