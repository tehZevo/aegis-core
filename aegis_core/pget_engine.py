import numpy as np
import tensorflow as tf

from ml_utils.keras import get_states, set_states, apply_regularization
from ml_utils.viz import viz_weights

from pget.pget import create_traces, update_traces, step_weights
from pget.pget import explore_continuous, explore_discrete, explore_multibinary
from pget.pget import categorical_crossentropy, binary_crossentropy

from .engine import RequestEngine

#TODO: make traces numpy arrays instead of tf variables
#TODO: squash (tanh) and scale in update instead of in the model itself?
# same with discrete softmax or whatever

class PGETEngine(RequestEngine):
  def __init__(self, modelpath, exploration_method="continuous", input_urls=[], save_interval=1000,
      #general config
      train=True, alt_trace_method=False, reward_transform=lambda x: x,
      reward_propagation=0, propagate_advantage=False,
      #hyperparameters
      epsilon=1e-7, advantage_clip=1, gamma=0.99, lr=1e-4, lambda_=0.9,
      regularization_scale=1e-4, optimizer="adam", noise=0.1):
    super().__init__(input_urls=input_urls)

    #TODO: populate hyperparameters from arguments
    self.do_train = train
    self.eps = epsilon
    self.advantage_clip = advantage_clip
    self.gamma = gamma
    self.lr = lr
    self.lambda_ = lambda_
    self.alt_trace_method = alt_trace_method
    self.regularization = regularization_scale * self.lr
    self.reward_transform = reward_transform
    self.noise = noise
    self.exploration = exploration_method.lower()
    #wow i actually forgot to change this from 0.9...
    self.reward_propagation = reward_propagation

    #TODO: support more optimizers by name... or by object
    self.optimizer = None if optimizer is None else tf.train.AdamOptimizer(self.lr)
    self.propagate_advantage = propagate_advantage
    self.save_interval = save_interval

    self.steps_since_save = 0

    self.modelpath = modelpath
    self.model = tf.keras.models.load_model(modelpath)

    self.input_shape = tuple(self.model.input_shape[1:])
    self.output_shape = tuple(self.model.output_shape[1:])

    if self.exploration == "discrete":
      self.loss = categorical_crossentropy
      explore_func = explore_discrete
    elif self.exploration == "multibinary":
      self.loss = binary_crossentropy
      explore_func = explore_multibinary
    elif self.exploration == "continuous":
      #TODO: try huber loss again?
      self.loss = tf.losses.mean_squared_error
      explore_func = explore_continuous
    else:
      raise ValueError("Unknown exploration method '{}'".format(exploration_method))
      
    self.explore = lambda x: explore_func(x, self.noise)

    self.traces = create_traces(self.model)

    self.reward_mean = 0
    self.reward_variance = 10

  def update(self, reward):
    transformed_reward = self.reward_transform(reward)
    advantage = self.calculate_advantage(transformed_reward)

    if self.do_train:
      self.train(advantage)

    prop_reward = advantage if self.propagate_advantage else reward

    #TODO: other reward propagation schemes?
    input_states = self.get_inputs(prop_reward * self.reward_propagation / len(self.input_urls))
    #fix Nones
    input_states = [np.zeros(self.input_shape) if x is None else x for x in input_states]
    #TODO: try running on each input, then averaging?
    input_state = np.mean(input_states, axis=0) #TODO: this will fail for discrete where a state is missing (set to 0s...) actually maybe not
    output_state = self.get_output_and_update_traces(input_state)

    self.steps_since_save += 1
    if self.do_train and self.steps_since_save > self.save_interval:
      print("saving model to {}".format(self.modelpath))
      self.model.save(self.modelpath)
      self.steps_since_save = 0

      viz_weights(self.model.get_weights(), self.modelpath + ".png")

    return output_state

  def get_output_and_update_traces(self, input_state):
    #housekeeping
    state = input_state.astype("float32")
    #save pre-step hidden state
    pre_step_state = get_states(self.model)
    #calc action from state
    action = self.model.predict(np.expand_dims(state, 0))[0]

    #apply noise to action
    action = self.explore(action)

    #early bail, TODO: include exploration?
    if not self.do_train:
      return action

    #calc gradient for modified action & update traces based on gradient
    update_traces(self.model, pre_step_state, self.traces,
      np.expand_dims(state, 0), np.expand_dims(action, 0), self.loss, lambda_=self.lambda_)

    return action

  def train(self, advantage):
    #step network in direction of trace gradient * lr * reward
    apply_regularization(self.model, self.regularization)
    step_weights(self.model, self.traces, self.lr, advantage, self.optimizer)

  def calculate_advantage(self, reward):
    #scale/clip reward
    delta_reward = reward - self.reward_mean
    advantage = delta_reward / (self.reward_variance + self.eps)
    if self.advantage_clip is not None:
      advantage = np.clip(advantage, -self.advantage_clip, self.advantage_clip)

    #update reward mean/variance
    self.reward_mean += delta_reward * (1 - self.gamma)
    self.reward_variance += (np.abs(delta_reward) - self.reward_variance) * (1 - self.gamma)

    return advantage
