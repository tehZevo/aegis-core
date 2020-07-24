import logging
import subprocess
import time
import sys

from aegis_core.reward_proxy import RewardProxy

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

delay = 1
port = 12400

processes = []

def allocate():
  global port
  #TODO: check if port is open (https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python)?
  p = port
  port += 1
  return p

def channel(port, route):
  return "{}/{}".format(port, route)

def rl_node(path, port, node_urls, input_shape, output_shape, name, discrete=False, reward_prop=0, niceness=1, logdir=None):
  #TODO: support multidim in/out shapes later
  node_urls = " ".join(map(str, node_urls))
  #input_shape = " ".join(map(str, input_shape))
  #output_shape = " ".join(map(str, output_shape))
  logdir = "-l {}".format(logdir) if logdir is not None else ""
  discrete = "-t discrete" if discrete else ""
  command = "python scripts/run_sb.py -u {} -p {} -i {} -o {} {} -f {} -s {} -n {} -r {} {}"
  command = command.format(node_urls, port, input_shape, output_shape, discrete, path, niceness, name, reward_prop, logdir)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

def env_node(env_name, port, node_urls, proxy=None, name=None, render=False, niceness=1):
  node_urls = " ".join(map(str, node_urls))
  name = env_name if name is None else name
  render = "--render" if render else ""
  proxy = ("-x {}".format(proxy)) if proxy is not None else ""
  command = 'python scripts/run_env.py -u {} -p {} -s {} -n {} {} -e "{}" {}'
  command = command.format(node_urls, port, niceness, name, render, env_name, proxy)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

def reward_proxy(port, node_urls, channels=[], niceness=1):
  #TODO: add args for clips/decay rates/scales?
  node_urls = " ".join(map(str, node_urls))
  channels = " ".join(map(str, channels))
  name = "reward_proxy"
  command = 'python scripts/run_reward_proxy.py -p {} -u {} -c {} -s {}'
  command = command.format(port, node_urls, channels, niceness)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

def curiosity(model_path, port, input_urls, action_url, name=None, niceness=1):
  #TODO: support train parameter
  input_urls = " ".join(map(str, input_urls))
  name = "curiosity" if name is None else name
  command = 'python scripts/run_curiosity.py -m {} -p {} -u {} -a {} -n {} -s {}'
  command = command.format(model_path, port, input_urls, action_url, name, niceness)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

#TODO: support image observation
def atari_gauntlet(port, node_urls, proxy=None, name=None, render=False, niceness=1, action_repeat=1, step_limit=10000):
  node_urls = " ".join(map(str, node_urls))
  name = "atari_gauntlet"
  #TODO: hardcoded render, observation type
  proxy = ("-x {}".format(proxy)) if proxy is not None else ""
  render = "--render" if render else ""
  command = 'python scripts/run_atari_gauntlet.py -u {} -p {} -s {} -n {} -k {} -l {} -o ram {} {}'
  command = command.format(node_urls, port, niceness, name, action_repeat, step_limit, render, proxy)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

def mnist_env(port, node_urls, proxy=None, blackout=False, niceness=1, step_limit=200):
  node_urls = " ".join(map(str, node_urls))
  name = "mnist"
  proxy = ("-x {}".format(proxy)) if proxy is not None else ""
  blackout = "--blackout" if blackout else ""
  command = 'python scripts/run_mnist_env.py -u {} -p {} -s {} -n {} -l {} {}'
  command = command.format(node_urls, port, niceness, name, step_limit, proxy)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

def keras_node(model_path, port, node_urls, niceness=1):
  node_urls = " ".join(map(str, node_urls))
  name = "keras node" #TODO: hardcoded
  command = 'python scripts/run_keras.py -f {} -u {} -p {} -s {}'
  command = command.format(model_path, node_urls, port, niceness)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

def discretizer(port, size, node_urls, niceness=1):
  node_urls = " ".join(map(str, node_urls))
  name = "discretizer" #TODO: hardcoded
  command = 'python -m aegis_core.discretizer_engine -u {} -s {} -p {} -n {}'
  command = command.format(node_urls, size, port, niceness)
  p = subprocess.Popen(command)
  processes.append((p, name, command))
  time.sleep(delay)

rp = 0 #reward propagation
nn = 2 #niceness

#allocate node ports
brain = allocate()              #12401
rewards = allocate()            #12402
lunar = allocate()              #12403
lunar_obs = allocate()          #12404
lunar_action = allocate()       #12405
brain_proj = allocate()         #12406
#lunar_discretizer = allocate()  #12405

nodes = [
  env_node("LunarLander-v2", lunar, [lunar_action], channel(rewards, "lunar"), niceness=nn)
  keras_node("models/alpha/lunar_obs_proj", lunar_obs, [lunar], niceness=nn)
  keras_node("models/alpha/lunar_action_proj", lunar_action, [brain], niceness=nn)
  keras_node("models/alpha/brain_proj", brain_proj, [brain], niceness=nn) #tanh activation 4/1/2020
]

processes = [subprocess.Popen(command) for command in nodes]

while True:
  for (p, name, command) in processes:
    if p.poll() != None:
      print("OH NO! <{}> ({}) has exited!".format(name, command))
      #TODO: kill all subprocesses

      for (p, name, command) in processes:
        p.terminate()

      sys.exit(1)
  time.sleep(10)
