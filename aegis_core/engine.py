import requests
import numpy as np
import re

class Engine:
  def __init__(self):
    self.input_shape = None
    self.output_shape = None

  def update(self, reward=0):
    return None

#TODO: move request engine stuff to its own file
def sanitize(url):
  if re.match(r"^\d+(/.*)?$", url):
    url = "localhost:" + url

  if not re.match(r"^https?://", url):
    url = "http://" + url

  return url

def request_input(url, reward):
  """warning: may return none"""
  try:
    r = requests.post(url, json=reward)
    r.raise_for_status()
    r = r.json()
    return np.array(r)
  except Exception as e:
    print(e)
    print("Error in get input, returning none")
    return None

class RequestEngine:
  def __init__(self, input_urls=[]):
    self.input_urls = [sanitize(url) for url in input_urls]

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
    return [request_input(url, reward) for url in self.input_urls]
