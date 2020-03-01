import threading
import time

import gym
from gym import spaces
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from tensorflow.keras.utils import to_categorical

from .flask_controller import ControllerResource
from .engine import RequestEngine

#WIP

#agent.train calls env reset/step
#so env has to set up flask server (controller?)
#env gets reward from requests to its flask server
#env gets action from agent, state comes from request or whatever..

#not a "controller", but needs `state` and `reward` variables
class AegisEnv(gym.Env):
  def __init__(self, obs_shape, action_shape, input_urls=[], discrete=False,
      niceness=0.1, port=8181, n_steps=None, reward_propagation=0):
    self.input_shape = obs_shape
    self.output_shape = action_shape
    self.niceness = niceness
    #TODO: remove dummy (requires controllerresource refactor?)
    self.engine = {"input_shape":self.input_shape, "output_shape":self.output_shape}
    #TODO: hardcoded low/highs
    self.observation_space = spaces.Box(shape=[obs_shape], low=-np.Inf, high=np.Inf)
    self.action_space = spaces.Discrete(action_shape) if discrete else spaces.Box(shape=[action_shape], low=-np.Inf, high=np.Inf)
    self.discrete = discrete
    self.n_steps = n_steps
    self.step_count = 0
    self.reward_propagation = reward_propagation
    #despite it being named "state", this is actually the output (ie action)
    #of the agent, to be passed to whichever agents request it down the line
    #do NOT return this from reset/step
    self.state = np.zeros(action_shape) #TODO: set to none to start?
    #reward for the next step
    self.reward = 0

    self.request_engine = RequestEngine(input_urls)
    self.request_engine.input_shape = self.input_shape #TODO: sigh, more patchwork
    self.start_server(port)

  def step(self, action):
    starttime = time.time()
    #aegis expects discrete actions to be represented by one-hot (for now)
    if self.discrete:
      action = to_categorical(action, self.action_space.n)
    #set state for other nodes to pick up
    self.state = action

    r = self.reward
    self.reward = 0
    obs = self.get_observation(r * self.reward_propagation);

    self.step_count += 1
    done = self.n_steps != None and (self.step_count >= self.n_steps)

    #sleep time equal to update time * niceness
    dt = time.time() - starttime
    if self.niceness >= 0:
      time.sleep(dt * self.niceness)
    else:
      time.sleep(-self.niceness)

    return obs, r, done, {}

  def get_observation(self, reward):
    #TODO: reward propagation scheme?
    inputs = self.request_engine.get_inputs(reward)
    inputs = [np.zeros(self.input_shape) if x is None else x for x in inputs]
    #TODO: other merge methods?
    return np.mean(inputs, axis=0)

  def reset(self):
    self.step_count = 0
    return self.get_observation(0)

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
