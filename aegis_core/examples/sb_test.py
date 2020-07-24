from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import gym
import numpy as np
from utils import DummyEnv

#TODO: support more obs spaces
obs_space = gym.spaces.Box(shape=args.input_shape, low=-np.Inf, high=np.Inf)
#TODO: support more action spaces
if args.discrete:
  action_space = gym.spaces.Discrete(args.output_shape[0])
else:
  action_space = gym.spaces.Box(shape=args.output_shape, low=-np.Inf, high=np.Inf)

#TODO: urls
env = AegisEnv(12400, "12399/observation", "12399/reward",
  lunar.observation_space.shape, [lunar.action_space.n],
  discrete=True, n_steps=1000)
env = DummyVecEnv([lambda: env])

#TODO: infer space types and shapes from saved agent?
#TODO: nminibatches/noptepochs?
model = PPO2("MlpPolicy", env, nminibatches=1)
model.save(args.path)



from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from aegis_core.aegis_env import AegisEnv

discrete = args.action_type == "discrete"
# Create environment
#TODO: may be able to access obs/action spaces from agent (would require setting env after load..)

env = DummyVecEnv([lambda: env])

#load model
model = PPO2.load(args.path, env, verbose=args.verbose, tensorboard_log=args.logdir)

#train
ep_counter = 0
while True:
  env.reset() #TODO: is this necessary?
  model.learn(total_timesteps=args.steps, reset_num_timesteps=False, tb_log_name=args.name)
  ep_counter += 1
  #TODO: actual step counter might be off because .learn might have different intervals
  print("Steps: {}".format(ep_counter * args.steps))
  model.save(args.path)
