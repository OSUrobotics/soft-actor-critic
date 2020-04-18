##############
## Name: Paresh Soni
## Refrence: Towards Data Science blog post
##############

##
#Imports
##
import math
import random

import gym
import matplotlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Cython.Shadow import inline
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from ounoise import OUNoise


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_action, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_action)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
		
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)
        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


class softactorcritic(object):
	def __init__(self, state_dim, action_dim, hidden_dim=256, n=5, discount=0.995, tau=0.0005):
		self.value_net = ValueNetwork(state_dim, hidden_dim, action_dim).to(device)
		self.target_value_net = ValueNetwork(state_dim, hidden_dim, action_dim).to(device)
		self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
		self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
		self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
		self.batch_size = 128
		value_lr = 3e-4
		soft_q_lr = 3e-4
		policy_lr = 3e-4
		self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
		self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
		self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
		#for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
   		#	target_param.data.copy_(param.data)
		self.discount = discount
		self.tau = tau
		self.n = n
		self.network_repl_freq = 10
		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		a = self.policy_net(state)[1].cpu().data.numpy().flatten()
		return a
		
	def train(self, replay_buffer, gamma=0.99, soft_tau=1e-2):
		state, action, next_state, reward, done = replay_buffer.sample_wo_expert()
		value_criterion = nn.MSELoss()
		soft_q_criterion1 = nn.MSELoss()
		soft_q_criterion2 = nn.MSELoss()
		state = torch.FloatTensor(state).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		action = torch.FloatTensor(action).to(device)
		reward = torch.FloatTensor(reward).to(device)
		done = torch.FloatTensor(np.float32(done)).to(device)
		predicted_q_value1 = self.soft_q_net1(state, action)
		predicted_q_value2 = self.soft_q_net2(state, action)
		predicted_value = self.value_net(state)
		new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)
		
		# Training Q Function
		target_value = self.target_value_net(next_state)
		target_q_value = reward + (1 - done) * gamma * target_value
		q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
		q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
		self.soft_q_optimizer1.zero_grad()
		q_value_loss1.backward()
		self.soft_q_optimizer1.step()
		self.soft_q_optimizer2.zero_grad()
		q_value_loss2.backward()
		self.soft_q_optimizer2.step()
		
		# Training Value Function
		predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
		target_value_func = predicted_new_q_value - log_prob
		value_loss = value_criterion(predicted_value, target_value_func.detach())
		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()
		
		# Training Policy Function
		policy_loss = (log_prob - predicted_new_q_value).mean()
		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()
		#for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
		#    target_param.data.copy_(
		#        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
		#    )
				
	def save(self, filename):
		torch.save(self.value_net.state_dict(), filename + "_value_net")
		torch.save(self.target_value_net.state_dict(), filename + "_target_value_net")
		torch.save(self.soft_q_net1.state_dict(), filename + "_soft_q_net1")
		torch.save(self.soft_q_net2.state_dict(), filename + "_soft_q_net2")
		torch.save(self.policy_net.state_dict(), filename + "_policy_net")
		torch.save(self.value_optimizer.state_dict(), filename + "_value_optimizer")
		torch.save(self.soft_q_optimizer1.state_dict(), filename + "_soft_q_optimizer1")
		torch.save(self.soft_q_optimizer2.state_dict(), filename + "_soft_q_optimizer2")
		torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")
		
	def load(self, filename):
		self.value_net.load_state_dict(torch.load(filename + "_value_net"))
		self.target_value_net.load_state_dict(torch.load(filename + "_target_value_net"))
		self.soft_q_net1.load_state_dict(torch.load(filename + "_soft_q_net1"))
		self.soft_q_net2.load_state_dict(torch.load(filename + "_soft_q_net2"))
		self.policy_net.load_state_dict(torch.load(filename + "_policy_net"))
		self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))	
		self.soft_q_optimizer1.load_state_dict(torch.load(filename + "_soft_q_optimizer1"))	
		self.soft_q_optimizer2.load_state_dict(torch.load(filename + "_soft_q_optimizer2"))	
		self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))		

