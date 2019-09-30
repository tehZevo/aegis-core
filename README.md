# Aegis

## An experiment in extensible transfer reinforcement learning

## How does it work?
The core protocol is very simple: you send a reward to a node, and it responds with its current state.
This seems to work so far for simple experiments (ex: a single agent and a single CartPole environment).
The concept of reward distribution and propagation through agents is under exploration.

Currently, the protocol is implemented using a single HTTP POST request.

## Terminology (subject to change):
### Node
* A single process that accepts rewards and returns states when requested
* Could be a reinforcement learning agent, a `Gym` environment, a static pretrained classifier/regressor, autoencoder, generative network, etc.
* Rewards need not be respected; they can simply be ignored or propagated elsewhere.

### Controller
* Accumulates rewards and distributes state upon request
* Calls an engine's update method to train and produce a new state

### Engine:
* Handles internal update logic such as accumulating input states, training a reinforcement learning agent, producing a new state, etc.

### Reward proxy:
* Accepts rewards which are then sent to multiple nodes at the same time

## TODO
* support environments with discrete observation spaces

* support environments with multidimensional observation and action spaces

* VAE engine
  * stores seen inputs and trains on them

* multiple input/output channels per node (mostly for VAE and pretrained SL models)

* static pretrained SL model engine (eg image classifier)

* monitoring / command API
  * connect to one node, it finds upstream connections?
    * how to handle downstream connections
  * config file for saving nodes?

* REST API for modifying node settings
  * connect/disconnect
  * learning rate, gamma/lambda, etc

* add "channels" and rescaling to reward proxy

* "PSET" engine
  * policy search with eligibility traces
  * uses parameter search instead of gradients (compared to PGET)
