import time
import numpy as np
import threading
import os
import math
import requests
import re

import asyncio

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

import logging

#TODO: make these static on class
DEFAULT_INPUT = "_default"
DEFAULT_OUTPUT = "" #for http route "/"

DEFAULT_NICENESS = 1.
DEFAULT_DELAY = 1./100

#TODO: automatically generate routes when set_output is called?

#TODO: accidentally reintroduced issue where recursive nodes will take more and more time..
# because the niceness time includes time taken to fetch inputs
# need to pre-fetch inputs...?

#for now, all inputs will be singlular

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
    self.cb_data = {}

  def get(self):
    #no hand holding here, only numpy arrays/scalars and None are allowed
    #receiving end is responsible for converting to a useful format and handling Nones/nulls
    data = self.node.output_states[self.output]
    if isinstance(data, (np.ndarray, np.generic)):
      data = data.tolist()
    data = jsonify(data)

    return data

#TODO: some kind of single parameter config (niceness, cors, callbacks?)
class AegisNode():
  def __init__(self, port, inputs=None, outputs=None):
    #TODO: add option for port-less node (no outputs)
    self.port = port
    self.niceness = DEFAULT_NICENESS
    #delay helps to prevent really light nodes from sucking cpu
    self.delay = DEFAULT_DELAY #TODO rename idk
    self.cors = True #TODO: remove param

    #time spent waiting on inputs
    self.input_time = 0

    #TODO: disables logging, might need a more graceful approach
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    if inputs is None:
      inputs = {}
    if isinstance(inputs, str):
      inputs = {DEFAULT_INPUT: inputs}
    if not isinstance(inputs, dict):
      raise "inputs should be a dict, or str, or None"

    inputs = {k: sanitize(v) for k, v in inputs.items()}

    self.inputs = inputs

    #TODO: don't create an output if no output..?
    if outputs is None:
      outputs = [DEFAULT_OUTPUT]
    if not isinstance(outputs, list):
      raise "outputs should be a list, or None" #TODO: or str?
    self.outputs = outputs
    self.output_states = {k: None for k in self.outputs}

    #setup server
    self.create_flask_server()
    self.setup_routes()
    self.start_flask_server()

  def set_niceness(self, niceness):
    self.niceness = niceness
    return self

  def set_delay(self, delay):
    self.delay = delay
    return self

  def create_flask_server(self):
    flask_app = Flask(__name__)
    flask_app.config['CORS_HEADERS'] = 'Content-Type'
    #NOTE: have to apply cors after defining resources?
    #use cors on aegis nodes, aint nobody got time for localhost errors
    CORS(flask_app, resources={r"/*": {"origins": "*"}})

    self.api = Api(flask_app)
    self.flask_app = flask_app

  def setup_routes(self):
    for output in self.outputs:
      print("/{}".format(output))
      self.api.add_resource(AegisResource, "/{}".format(output), endpoint=output, resource_class_kwargs={"node": self, "output": output})

  def start_flask_server(self):
    def dedotated_wam():
      self.flask_app.run(debug=False, threaded=True, port=self.port)

    self.app_thread = threading.Thread(target=dedotated_wam)
    self.app_thread.daemon = True
    self.app_thread.start()

  def has_input(self, channel):
    return channel in self.inputs

  def has_output(self, channel):
    return channel in self.outputs

  def get_input(self, channel=None, shape=None):
    """Returns input from the given channel
    if shape is provided, will return np.zeros(shape) if failed/none
    """
    input_start_time = time.time()
    channel = DEFAULT_INPUT if channel is None else channel
    if not self.has_input(channel):
      raise Exception("Channel '{}' not present in inputs".format(channel))

    data = None
    try:
      data = requests.get(self.inputs[channel])
      data.raise_for_status()
      data = data.json()
      if data is not None:
        data = np.array(data)
    except:
      print("Error when getting input '{}'".format(channel))

    if data is None and shape is not None:
      data = np.zeros(shape)

    #add time spent waiting on input
    self.input_time += time.time() - input_start_time

    return data

  async def start(self):
    while True:
      #call update
      t = time.time()
      self.update()
      dt = time.time() - t

      #print(type(self).__name__, dt) #print update time

      #subtract time spent waiting on inputs, so delay doesnt increase forever
      #TODO: might have to reduce niceness to like 0.9 too?
      dt -= self.input_time
      self.input_time = 0

      #sleep according to niceness and delay
      #time.sleep(dt * self.niceness)
      await asyncio.sleep(dt * self.niceness)
      #time.sleep(self.delay)
      await asyncio.sleep(self.delay)

  def update(self):
    #override me
    pass

  def set_output(self, state, channel=None):
    """Sets the output state of the given channel"""
    channel = DEFAULT_OUTPUT if channel is None else channel
    if not self.has_output(channel):
      raise Exception("Channel '{}' not present in outputs".format(channel))
    self.output_states[channel] = state
