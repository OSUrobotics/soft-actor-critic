import numpy as np
import torch
import gym
import argparse
import os
import softactorcritic
import utils
import TD3
import OurDDPG
import DDPG
import DDPGfD
import pdb
from tensorboardX import SummaryWriter
from ounoise import OUNoise
import pickle
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	avg_reward = 0.0
	
	for i in range(eval_episodes):
		eval_env = gym.make(env_name)
		state, done = eval_env.reset(), False
		cumulative_reward = 0
		
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			cumulative_reward += reward
			
	avg_reward /= eval_episodes
	print("---------------------------------------")
	# print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print(f"Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="softactor")				# Policy name
	parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0") # OpenAI gym environment name
	parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=100, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=100, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)		# Max time steps to run environment for
	parser.add_argument("--max_episode", default=20000, type=int)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=250, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.995, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.0005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.01, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.05, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--tensorboardindex", default="new")	# tensorboard log name
	parser.add_argument("--model", default=1, type=int)	# save model index
	parser.add_argument("--pre_replay_episode", default=100, type=int)	# Number of episode for loading expert trajectories
	parser.add_argument("--saving_dir", default="new")	# Number of episode for loading expert trajectories
	
	args = parser.parse_args()
	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print("---------------------------------------")
	print("Settings: {file_name}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env_name)
	
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	max_action_trained = env.action_space.high # a vector of max actions

	kwargs = {
		"state_dim": state_dim, 
		"action_dim": action_dim, 

		"discount": args.discount,
		"tau": args.tau,
		# "trained_model": "data_cube_5_trained_model_10_07_19_1749.pt"		
	}

	# Initialize policy
	if args.policy_name == "TD3": 
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy_name == "OurDDPG": 
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy_name == "DDPG": 		
		policy = DDPG.DDPG(**kwargs)
	elif args.policy_name == "DDPGfD":
		policy = DDPGfD.DDPGfD(**kwargs)
	elif args.policy_name == "softactor":
		policy = softactorcritic.softactorcritic(**kwargs)

	# Initialize replay buffer with expert demo
	print("----Generating {} expert episodes----".format(args.pre_replay_episode))
	from expert_data import generate_Data, store_saved_data_into_replay, GenerateExpertPID_JointVel

	# trained policy
	#policy.load("./policies/reward_all/softactor_kinovaGrip_10_22_19_2151")

	# old pid control
	replay_buffer = utils.ReplayBuffer_episode(state_dim, action_dim, env._max_episode_steps, args.pre_replay_episode, args.max_episode)

	evaluations = []
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Check and create directory
	saving_dir = "./policies/" + args.saving_dir
	if not os.path.isdir(saving_dir):
		os.mkdir(saving_dir)
	model_save_path = saving_dir + "/softactor_kinovaGrip_{}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))

	# Initialize OU noise
	noise = OUNoise(4)
	noise.reset()
	expl_noise = OUNoise(4, sigma=0.001)
	expl_noise.reset()

	# Initialize SummaryWriter 
	writer = SummaryWriter(logdir="./kinova_gripper_strategy/{}_{}/".format(args.policy_name, args.tensorboardindex))

	print("---- RL training in process ----")
	for t in range(int(args.max_episode)):
		env = gym.make(args.env_name)	
		episode_num += 1
		state, done = env.reset(), False
		noise.reset()
		expl_noise.reset()
		episode_reward = 0
		while not done:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			# Perform action
			next_state, reward, done, _ = env.step(action)
			env.render()
			done_bool = float(done) # if episode_timesteps < env._max_episode_steps else 0
			replay_buffer.add_wo_expert(state, action, next_state, reward, done_bool)
			state = next_state
			episode_reward += reward

		# Train agent after collecting sufficient data:
		if episode_num > 10:
			for learning in range(100):
				policy.train(replay_buffer)
		# 		actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train(replay_buffer)
		#
		# # Publishing loss
		# 	writer.add_scalar("Episode reward", episode_reward, episode_num)
		# 	writer.add_scalar("Actor loss", actor_loss, episode_num)
		# 	writer.add_scalar("Critic loss", critic_loss, episode_num)
		# 	writer.add_scalar("Critic L1loss", critic_L1loss, episode_num)
		# 	writer.add_scalar("Critic LNloss", critic_LNloss, episode_num)
		# replay_buffer.add_episode(0)

		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env_name, args.seed))
			np.save("./results/%s" % (file_name), evaluations)
			print()

	print("Saving into {}".format(model_save_path))
	policy.save(model_save_path)
