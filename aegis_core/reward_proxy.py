import numpy as np

from .engine import RequestEngine

class RewardProxy(RequestEngine):
  def __init__(self, node_urls=[]):
    super().__init__(input_urls=node_urls)
    #TODO: problem, if we want multiple "channels", then our flask controller needs multiple routes..
    #for controller >_>
    self.input_size = 0;
    self.output_size = 0;

  def update(self, reward):
    #distribute reward, discard tensors.. who needs them anyway?
    self.get_inputs(reward);

    return np.zeros([0]);
