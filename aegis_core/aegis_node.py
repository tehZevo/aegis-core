import time
import numpy as np
import threading
import os
import math
import requests
import re

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

DEFAULT_INPUT = "_default"
DEFAULT_OUTPUT = "" #for http route "/"

#TODO: automatically generate routes when set_state is called?
#TODO: separate rewards per url?

def sanitize(url):
  if re.match(r"^\d+(/.*)?$", url):
    #https://hackernoon.com/how-changing-localhost-to-127-0-0-1-sped-up-my-test-suite-by-1-800-8143ce770736
    #url = "localhost:" + url
    url = "127.0.0.1:" + url

  if not re.match(r"^https?://", url):
    url = "http://" + url

  return url

class AegisResource(Resource):
  def __init__(self, node, output):
    self.node = node
    self.output = output
    #TODO: use output (store rewards correctly)
    self.cb_data = {}

  def post(self):
    reward = request.get_json(force=True)
    if reward is None:
      #TODO: treat null/none/empty reward as 0?
      raise ValueError("Reward was none!")
      os._exit(1)
    if math.isnan(reward):
      raise ValueError("Reward was nan!")
      os._exit(1)

    #store reward in respective channel
    self.node.received_reward_buffer[self.output] += reward

    #no hand holding here, only numpy arrays/scalars and None are allowed
    #receiving end is responsible for converting to a useful format and handling Nones/nulls
    data = self.node.output_states[self.output]
    if isinstance(data, (np.ndarray, np.generic)):
      data = data.tolist()
    data = jsonify(data)

    return data

#TODO: some kind of single parameter config (niceness, cors, callbacks?)
class AegisNode():
  def __init__(self, port, inputs=None, outputs=None, niceness=1, cors=True, callbacks=None):
    self.port = port
    self.niceness = niceness
    self.cors = cors
    self.callbacks = [] if callbacks is None else callbacks

    if inputs is None:
      inputs = {} #:^)
    if isinstance(inputs, str):
      inputs = [inputs]
    if isinstance(inputs, list):
      inputs = {DEFAULT_INPUT: inputs}
    if isinstance(inputs, dict):
      inputs = {k: v if isinstance(v, list) else [v] for k, v in inputs.items()}
    if not isinstance(inputs, dict):
      raise "inputs should be a list, dict, or str"

    inputs = {k: [sanitize(x) for x in v] for k, v in inputs.items()}
    self.inputs = inputs
    self.send_rewards = {k: 0 for k, v in self.inputs.items()}

    if outputs is None:
      outputs = [DEFAULT_OUTPUT]
    if not isinstance(outputs, list):
      raise "outputs should be a list, or None"
    self.outputs = outputs
    self.received_rewards = {k: 0 for k in self.outputs}
    self.received_reward_buffer = {k: 0 for k in self.outputs}
    self.output_states = {k: None for k in self.outputs}

    #setup server
    self.create_flask_server()
    self.setup_routes()
    self.start_flask_server()

  def create_flask_server(self):
    flask_app = Flask(__name__)
    flask_app.config['CORS_HEADERS'] = 'Content-Type'
    #NOTE: have to apply cors after defining resources?
    if self.cors:
      print("Enabling CORS")
      CORS(flask_app, resources={r"/*": {"origins": "*"}})

    self.api = Api(flask_app)
    self.flask_app = flask_app

  def setup_routes(self):
    for output in self.outputs:
      self.api.add_resource(AegisResource, "/{}".format(output), resource_class_kwargs={"node": self, "output": output})

  def start_flask_server(self):
    def dedotated_wam():
      self.flask_app.run(debug=False, threaded=True, port=self.port)

    self.app_thread = threading.Thread(target=dedotated_wam)
    self.app_thread.daemon = True
    self.app_thread.start()

  def clear_send_rewards(self):
    self.send_rewards = {k: 0 for k, v in self.inputs.items()}

  def flip_received_rewards(self):
    self.received_rewards = self.received_reward_buffer.copy()
    self.received_reward_buffer = {k: 0 for k in self.outputs}

  def pre_update(self):
    self.fetch_inputs()
    self.flip_received_rewards()

  def post_update(self):
    for cb in self.callbacks:
      cb(self.cb_data)

  def set_callback_data(self, cb_data):
    self.cb_data = cb_data

  def fetch_inputs(self):
    #TODO: multithread maybe?
    self.input_states = {k: [None for _ in v] for k, v in self.inputs.items()}
    for input_name, urls in self.inputs.items():
      reward = self.send_rewards[input_name]
      self.input_states[input_name] = []
      for url in urls:
        try:
          data = requests.post(url, json=reward)
          data.raise_for_status()
          data = data.json()
          #TODO: none check here?
          #TODO: multithreading should use indexes instead of .append
          self.input_states[input_name].append(np.array(data))
        except Exception as e:
          print(e)
          print("Error in get input '{}', returning none".format(input_name))
    self.clear_send_rewards()

  def start(self):
    while True:
      self.internal_update()

  def internal_update(self):
    #TODO: clear output states?
    #get inputs, send rewards, flip received reward buffer
    self.pre_update()

    #call update
    t = time.time()
    self.update()
    dt = time.time() - t

    self.post_update()

    #sleep according to niceness
    if self.niceness >= 0:
      time.sleep(dt * self.niceness)
    else:
      time.sleep(-self.niceness)

  def update(self):
    #override me
    pass

  def set_state(self, state, channel=None):
    """Sets the output state of the given channel"""
    channel = DEFAULT_OUTPUT if channel is None else channel
    if channel not in self.outputs:
      raise Exception("Channel '{}' not present in outputs".format(channel))
    self.output_states[channel] = state

  def get_input(self, channel=None):
    """Returns the input at the given channel"""
    channel = DEFAULT_INPUT if channel is None else channel
    if channel not in self.inputs:
      raise Exception("Channel '{}' not present in inputs".format(channel))
    return self.input_states[channel]

  def give_reward(self, reward, channel=None):
    """Adds reward to the given input channel, which will be sent next step"""
    channel = DEFAULT_INPUT if channel is None else channel
    if channel not in self.inputs:
      raise Exception("Channel '{}' not present in inputs".format(channel))
    self.send_rewards[channel] += reward

  def get_reward(self, channel=None):
    """Returns the cumulative reward in the given output channel since the last step"""
    channel = DEFAULT_OUTPUT if channel is None else channel
    if channel not in self.outputs:
      raise Exception("Channel '{}' not present in outputs".format(channel))
    return self.received_rewards[channel]
