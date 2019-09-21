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
    """warning: may return none"""

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
    """warning: may return nones"""
    return [self.get_single_input(url, reward) for url in self.input_urls]
