import threading
import time

import gym
from gym import spaces
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from .flask_controller import ControllerResource
from .engine import RequestEngine

#WIP

#agent.train calls env reset/step
#so env has to set up flask server (controller?)
#env gets reward from requests to its flask server
#env gets action from agent, state comes from request or whatever..

#not a "controller", but needs `state` and `reward` variables
#TODO: reward propagation...?
class AegisEnv(gym.Env):
  def __init__(self, obs_size, action_size, input_urls=[], discrete=False, sleep=0.1, port=8181):
    self.input_size = obs_size
    self.output_size = action_size
    self.sleep = sleep
    #TODO: remove dummy (requires controllerresource refactor?)
    self.engine = {"input_size":self.input_size, "output_size":self.output_size}
    #TODO: hardcoded low/highs
    self.observation_space = spaces.Box(shape=[obs_size], low=-np.Inf, high=np.Inf)
    self.action_space = spaces.Discrete(action_size) if discrete else spaces.Box(shape=[action_size], low=-np.Inf, high=np.Inf)
    self.discrete = discrete
    #despite it being named "state", this is actually the output (ie action)
    #of the agent, to be passed to whichever agents request it down the line
    #do NOT return this from reset/step
    self.state = np.zeros([action_size])
    #reward for the next step
    self.reward = 0

    self.request_engine = RequestEngine(input_urls)
    self.request_engine.input_size = self.input_size #TODO: sigh, more patchwork
    self.start_server(port)

  def step(self, action):
    #TODO: how to handle discrete actions? to_categorical?
    #set state for other nodes to pick up
    self.state = action

    time.sleep(self.sleep) #TODO: move sleep? idk
    r = self.reward
    self.reward = 0
    #accumulate states from other nodes
    inputs = self.request_engine.get_inputs(r)
    #TODO: other merge methods?
    inputs = np.mean(inputs, axis=0)

    return inputs, r, False, {}

  def reset(self):
    inputs = self.request_engine.get_inputs(0)
    #TODO: other merge methods?
    inputs = np.mean(inputs, axis=0)
    return inputs

  #jacked from aegis -> flask_controller.py
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
