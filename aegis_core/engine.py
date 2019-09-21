import requests
import numpy as np

class Engine:
  def __init__(self):
    pass

  def update(self, reward):
    return None

class RequestEngine:
  def __init__(self, input_urls=[]):
    self.input_urls = input_urls

  def get_single_input(self, url, reward):
    try:
      #TODO: prefetch
      r = requests.post(url, json=reward)
      r.raise_for_status()
      r = r.json()
      return np.array(r)
    except Exception as e:
      print(e)
      print("error in get input, returning none")
      return None

  def get_inputs(self, reward):
    if len(self.input_urls) == 0:
      print("no urls, returning zeros")
      return np.zeros([1, self.input_size]) #1 for url dim

    inputs = [self.get_single_input(url, reward) for url in self.input_urls]
    #fix nones TODO: remove?
    #inputs = [np.zeros([self.input_size]) if x is None else x for x in inputs]
    return inputs
