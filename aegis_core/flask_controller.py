import numpy as np
import threading
import os
import math

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from .controller import Controller

class ControllerResource(Resource):
  def __init__(self, controller):
    self.controller = controller

  def post(self):
    reward = request.get_json(force=True)
    if reward is None:
      raise ValueError("Reward was none!")
      os._exit(1)
    if math.isnan(reward):
      raise ValueError("Reward was nan!")
      os._exit(1)

    self.controller.reward += reward
    data = self.controller.state
    if data is None:
      data = np.zeros(self.controller.engine.output_shape)
    if np.isnan(data).any():
      raise ValueError("NANS IN OUTPUT! {}".format(data))
      os._exit(1)
    data = jsonify(data.tolist())

    return data

class FlaskController(Controller):
  def __init__(self, engine, niceness=1, port=8181, autostart=True, cors=True):
    super().__init__(engine, niceness=niceness)
    self.cors = cors
    self.start_server(port)

    if autostart:
      self.loop()

  def start_server(self, port):
    flask_app = Flask(__name__)
    flask_app.config['CORS_HEADERS'] = 'Content-Type'
    #NOTE: have to apply cors after defining resources?
    if self.cors:
      print("Enabling CORS")
      CORS(flask_app, resources={r"/*": {"origins": "*"}})

    api = Api(flask_app)

    api.add_resource(ControllerResource, "/", resource_class_kwargs={"controller": self})

    self.flask_app = flask_app

    def dedotated_wam():
      self.flask_app.run(debug=False, threaded=True, port=port)

    self.app_thread = threading.Thread(target=dedotated_wam)
    self.app_thread.daemon = True
    self.app_thread.start()
