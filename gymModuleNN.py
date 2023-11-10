import gym
import pybullet_envs
import gymBasicModule as gbm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

##################################################
#
# What does Module "gymModuleNN" have :
# 
#  1. ActorNN(Actor Network class)
#  2. CriticNN(Critic Network class)
#  3. TargetActor(Target Actor Network class)
#  4. TargetCritic(Target Critic Network class)
#  5. ACAgent(Actor-Critic Agent class)
#  6. TD3Agent(TD3 Agent class)
#  7. TD3AgentABL1(TD3 Agent class without Target Actor and Target Critic)
#  8. TD3AgentABL2(TD3 Agent class without Target Policy Smoothing Regularization)
#  9. TD3AgentABL3(TD3 Agent class without Delayed Policy Update)
# 10. TD3AgentABL4(TD3 Agent class without Clipped Double Q-Learning)
#
##################################################

class ActorNN(nn.Module):

	def __init__(self, envName):
		super(ActorNN, self).__init__()

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.fc1 = nn.Linear(self.envDict.getStateDim(), 256)
		self.fc2 = nn.Linear(256, self.envDict.getActionDim())

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = torch.FloatTensor([(self.envDict.getActionRange(0)[1] - self.envDict.getActionRange(0)[0]) / 2]) * torch.tanh(self.fc2(x))
		return x

class CriticNN(nn.Module):

	def __init__(self, envName):
		super(CriticNN, self).__init__()

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.fc1 = nn.Linear(self.envDict.getActionDim() + self.envDict.getStateDim(), 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return x

class TargetActor():

	def __init__(self, actor, tau, invLen):
		self.actor = actor
		tmp = []
		for param in actor.parameters():
			tmp.append(param.data)
		self.w1 = tmp[0].clone().detach()
		self.b1 = tmp[1].clone().detach()
		self.w2 = tmp[2].clone().detach()
		self.b2 = tmp[3].clone().detach()

		self.tau = tau
		self.invLen = invLen

	def propagation(self, x):
		x = torch.add(torch.matmul(x, torch.transpose(self.w1, 0, 1)), self.b1)
		x = F.relu(x)
		x = torch.add(torch.matmul(x, torch.transpose(self.w2, 0, 1)), self.b2)
		x = torch.mul(torch.FloatTensor([self.invLen]), torch.tanh(x))
		return x

	def renewal(self):
		self.w1 = torch.mul(self.w1, 1 - self.tau)
		self.b1 = torch.mul(self.b1, 1 - self.tau)
		self.w2 = torch.mul(self.w2, 1 - self.tau)
		self.b2 = torch.mul(self.b2, 1 - self.tau)

		tmp = []
		for param in self.actor.parameters():
			tmp.append(param.data)
		self.w1 = torch.add(self.w1, tmp[0].clone().detach(), alpha=self.tau)
		self.b1 = torch.add(self.b1, tmp[1].clone().detach(), alpha=self.tau)
		self.w2 = torch.add(self.w2, tmp[2].clone().detach(), alpha=self.tau)
		self.b2 = torch.add(self.b2, tmp[3].clone().detach(), alpha=self.tau)

	def saveParameters(self, pathPrefix):
		torch.save(self.w1, pathPrefix + "_001.pth")
		torch.save(self.b1, pathPrefix + "_002.pth")
		torch.save(self.w2, pathPrefix + "_003.pth")
		torch.save(self.b2, pathPrefix + "_004.pth")

	def loadParameters(self, pathPrefix):
		self.w1 = torch.load(pathPrefix + "_001.pth")
		self.b1 = torch.load(pathPrefix + "_002.pth")
		self.w2 = torch.load(pathPrefix + "_003.pth")
		self.b2 = torch.load(pathPrefix + "_004.pth")

class TargetCritic():

	def __init__(self, critic, tau):
		self.critic = critic
		tmp = []
		for param in critic.parameters():
			tmp.append(param.data)
		self.w1 = tmp[0].clone().detach()
		self.b1 = tmp[1].clone().detach()
		self.w2 = tmp[2].clone().detach()
		self.b2 = tmp[3].clone().detach()
		self.w3 = tmp[4].clone().detach()
		self.b3 = tmp[5].clone().detach()

		self.tau = tau

	def propagation(self, x):
		x = torch.add(torch.matmul(x, torch.transpose(self.w1, 0, 1)), self.b1)
		x = F.relu(x)
		x = torch.add(torch.matmul(x, torch.transpose(self.w2, 0, 1)), self.b2)
		x = F.relu(x)
		x = torch.add(torch.matmul(x, torch.transpose(self.w3, 0, 1)), self.b3)
		return x

	def renewal(self):
		self.w1 = torch.mul(self.w1, 1 - self.tau)
		self.b1 = torch.mul(self.b1, 1 - self.tau)
		self.w2 = torch.mul(self.w2, 1 - self.tau)
		self.b2 = torch.mul(self.b2, 1 - self.tau)
		self.w3 = torch.mul(self.w3, 1 - self.tau)
		self.b3 = torch.mul(self.b3, 1 - self.tau)

		tmp = []
		for param in self.critic.parameters():
			tmp.append(param.data)
		self.w1 = torch.add(self.w1, tmp[0].clone().detach(), alpha=self.tau)
		self.b1 = torch.add(self.b1, tmp[1].clone().detach(), alpha=self.tau)
		self.w2 = torch.add(self.w2, tmp[2].clone().detach(), alpha=self.tau)
		self.b2 = torch.add(self.b2, tmp[3].clone().detach(), alpha=self.tau)
		self.w3 = torch.add(self.w3, tmp[4].clone().detach(), alpha=self.tau)
		self.b3 = torch.add(self.b3, tmp[5].clone().detach(), alpha=self.tau)

	def saveParameters(self, pathPrefix):
		torch.save(self.w1, pathPrefix + "_001.pth")
		torch.save(self.b1, pathPrefix + "_002.pth")
		torch.save(self.w2, pathPrefix + "_003.pth")
		torch.save(self.b2, pathPrefix + "_004.pth")
		torch.save(self.w3, pathPrefix + "_005.pth")
		torch.save(self.b3, pathPrefix + "_006.pth")

	def loadParameters(self, pathPrefix):
		self.w1 = torch.load(pathPrefix + "_001.pth")
		self.b1 = torch.load(pathPrefix + "_002.pth")
		self.w2 = torch.load(pathPrefix + "_003.pth")
		self.b2 = torch.load(pathPrefix + "_004.pth")
		self.w3 = torch.load(pathPrefix + "_005.pth")
		self.b3 = torch.load(pathPrefix + "_006.pth")

class ACAgent(gbm.Agent):

	def __init__(self, envName, free_exploration_step, gamma, alpha, noise):
		self.actor = ActorNN(envName)
		self.critic = CriticNN(envName)

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.acOptimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
		self.crOptimizer = torch.optim.Adam(self.critic.parameters(), lr=alpha)

		self.mse_loss = nn.MSELoss(reduction='mean')

		self.free_exploration_step = free_exploration_step
		self.gamma = gamma
		self.alpha = alpha
		self.noise = noise

		self.train_step = 0

	def save_models(self, pathPrefix):
		torch.save(self.actor, pathPrefix +"_AC_AC.pth")
		torch.save(self.critic, pathPrefix +"_AC_CR.pth")

	def load_models(self, pathPrefix):
		self.actor = torch.load(pathPrefix +"_AC_AC.pth")
		self.critic = torch.load(pathPrefix +"_AC_CR.pth")

	def select_action(self, state):
		actorInput = torch.FloatTensor(state)
		actorOutput = self.actor(actorInput)
		result = []
		for i in range(actorOutput.size(0)):
			result.append(actorOutput[i].item())
		self.train_step = self.train_step + 1
		return result

	def select_exploratory_action(self, state, freeExplFlag=True):
		result = []
		actDim = self.envDict.getActionDim()
		actMin = self.envDict.getActionRange(0)[0]
		actMax = self.envDict.getActionRange(0)[1]
		if freeExplFlag and (self.train_step < self.free_exploration_step):
			for i in range(actDim):
				tmp = np.random.rand() - 0.5
				tmp = tmp * (actMax - actMin)
				result.append(tmp)
		else:
			actorInput = torch.FloatTensor(state)
			actorOutput = self.actor(actorInput)
			for i in range(actDim):
				selectNoise = 0.5 * (actMax - actMin) * self.noise * np.random.randn()
				tmp = actorOutput[i].item() + selectNoise
				if tmp > actMax:
					tmp = actMax
				if tmp < actMin:
					tmp = actMin
				result.append(tmp)
		self.train_step = self.train_step + 1
		return result

	def trainExpReplay(self, replayBuffer, batch_size):

		data = replayBuffer.sample(batch_size)
		
		# make train data
		states_cr = torch.FloatTensor(data[0])
		actions_cr = torch.FloatTensor(data[1])
		next_states_cr1 = torch.FloatTensor(data[2])
		next_states_cr2 = torch.FloatTensor(data[2])
		rewards_cr = torch.FloatTensor(data[3])
		rewards_cr = torch.unsqueeze(rewards_cr, dim=1)

		tmp = []
		dones = data[4]
		for i in range(len(dones)):
			if dones[i]:
				tmp.append(0)
			else: tmp.append(self.gamma)
		dones_cr = torch.FloatTensor(tmp)
		dones_cr = torch.unsqueeze(dones_cr, dim=1)

		# calculate target value(critic)
		criticInputs1 = torch.cat([states_cr, actions_cr], dim=1)
		predictions = self.critic(criticInputs1)
		criticInputs2 = torch.cat([next_states_cr1, self.actor(next_states_cr2)], dim=1)
		ans_nextStep = torch.mul(dones_cr, self.critic(criticInputs2))
		answers = torch.add(rewards_cr, ans_nextStep)
		cost = self.mse_loss(predictions, answers.detach())

		#renewal(critic)
		self.crOptimizer.zero_grad()
		self.acOptimizer.zero_grad()
		cost.backward()
		self.crOptimizer.step()

		states_ac1 = torch.FloatTensor(data[0])
		states_ac2 = torch.FloatTensor(data[0])

		#calculate target value(actor)
		criticInputs3 = torch.cat([states_ac1, self.actor(states_ac2)], dim=1)
		criticResult = self.critic(criticInputs3)
		loss = torch.neg(torch.mean(criticResult))

		#renewal(actor)
		self.crOptimizer.zero_grad()
		self.acOptimizer.zero_grad()
		loss.backward()
		self.acOptimizer.step()

class TD3Agent(ACAgent):

	def __init__(self, envName, free_exploration_step, gamma, alpha, noise, tau, targetDeviation, clipWidth, delay):
		self.actor = ActorNN(envName)
		self.critic1 = CriticNN(envName)
		self.critic2 = CriticNN(envName)

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.acOptimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
		self.cr1Optimizer = torch.optim.Adam(self.critic1.parameters(), lr=alpha)
		self.cr2Optimizer = torch.optim.Adam(self.critic2.parameters(), lr=alpha)

		self.targetActor = TargetActor(self.actor, tau, (self.envDict.getActionRange(0)[1] - self.envDict.getActionRange(0)[0]) / 2)
		self.targetCritic1 = TargetCritic(self.critic1, tau)
		self.targetCritic2 = TargetCritic(self.critic2, tau)

		self.mse_loss = nn.MSELoss(reduction='mean')

		self.free_exploration_step = free_exploration_step
		self.gamma = gamma
		self.alpha = alpha
		self.noise = noise

		self.tau = tau
		self.targetDeviation = targetDeviation
		self.clipWidth = clipWidth
		self.delay = delay

		self.train_step = 0
		self.current_step = 0

	def save_models(self, pathPrefix):
		torch.save(self.actor, pathPrefix +"_TD3_AC.pth")
		torch.save(self.critic1, pathPrefix + "_TD3_CR1.pth")
		torch.save(self.critic2, pathPrefix + "_TD3_CR2.pth")
		self.targetActor.saveParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic1.saveParameters(pathPrefix + "_TD3_TCR1")
		self.targetCritic2.saveParameters(pathPrefix + "_TD3_TCR2")

	def load_models(self, pathPrefix):
		self.actor = torch.load(pathPrefix +"_TD3_AC.pth")
		self.critic1 = torch.load(pathPrefix + "_TD3_CR1.pth")
		self.critic2 = torch.load(pathPrefix + "_TD3_CR2.pth")
		self.targetActor.loadParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic1.loadParameters(pathPrefix + "_TD3_TCR1")
		self.targetCritic2.loadParameters(pathPrefix + "_TD3_TCR2")

	def trainExpReplay(self, replayBuffer, batch_size):
		data = replayBuffer.sample(batch_size)

		smoothingNoise = []
		actDim = self.envDict.getActionDim()
		for j in range(batch_size):
			tmp = np.clip(np.random.normal(0, self.targetDeviation, actDim), -1 * self.clipWidth, self.clipWidth).tolist()
			smoothingNoise.append(tmp)

		# make train data
		states_cr = torch.FloatTensor(data[0])
		actions_cr = torch.FloatTensor(data[1])
		next_states_cr = torch.FloatTensor(data[2])
		rewards_cr = torch.FloatTensor(data[3])
		rewards_cr = torch.unsqueeze(rewards_cr, dim=1)
		tmp = []
		dones = data[4]
		for i in range(len(dones)):
			if dones[i]:
				tmp.append(0)
			else: tmp.append(self.gamma)
		dones_cr = torch.FloatTensor(tmp)
		dones_cr = torch.unsqueeze(dones_cr, dim=1)

		smoothingNoise_cr = torch.FloatTensor(smoothingNoise)
		#if self.envDict.getActionDim() == 1:
		#	smoothingNoise_cr = torch.unsqueeze(smoothingNoise_cr, dim=1)

		# calculate target value = delta_t(critic)
		smoothingAction_cr = torch.add(self.targetActor.propagation(next_states_cr), smoothingNoise_cr)
		smoothingAction_cr = torch.clamp(smoothingAction_cr, min=self.envDict.getActionRange(0)[0], max=self.envDict.getActionRange(0)[1])
		criticInput_ans_cr = torch.cat([next_states_cr, smoothingAction_cr], dim=1)
		ans1_nextStep = self.targetCritic1.propagation(criticInput_ans_cr)
		ans2_nextStep = self.targetCritic2.propagation(criticInput_ans_cr)
		ans_nextStep = torch.minimum(ans1_nextStep, ans2_nextStep)
		answers = torch.add(rewards_cr, torch.mul(dones_cr, ans_nextStep))

		#renewal(critic1)
		criticInputs1 = torch.cat([states_cr, actions_cr], dim=1)
		predictions1 = self.critic1(criticInputs1)
		cost1 = self.mse_loss(predictions1, answers.detach())
		self.cr1Optimizer.zero_grad()
		cost1.backward()
		self.cr1Optimizer.step()

		#renewal(critic2)
		criticInputs2 = torch.cat([states_cr, actions_cr], dim=1)
		predictions2 = self.critic2(criticInputs2)
		cost2 = self.mse_loss(predictions2, answers.detach())
		self.cr2Optimizer.zero_grad()
		cost2.backward()
		self.cr2Optimizer.step()

		if self.current_step == self.delay:
			#calculate target value(actor)
			states_ac = torch.FloatTensor(data[0])
			criticInputs = torch.cat([states_ac, self.actor(states_ac)], dim=1)
			criticResult = self.critic1(criticInputs)
			loss = torch.neg(torch.mean(criticResult))

			#renewal(actor)
			self.acOptimizer.zero_grad()
			loss.backward()
			self.acOptimizer.step()

			#renewal(targetActor, targetCritic)
			self.targetActor.renewal()
			self.targetCritic1.renewal()
			self.targetCritic2.renewal()

			#clear current step buffer
			self.current_step = 0
		else:
			self.current_step = self.current_step + 1

class TD3AgentABL1(ACAgent):
	# without target critic, target actor

	def __init__(self, envName, free_exploration_step, gamma, alpha, noise, tau, targetDeviation, clipWidth, delay):
		self.actor = ActorNN(envName)
		self.critic1 = CriticNN(envName)
		self.critic2 = CriticNN(envName)

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.acOptimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
		self.cr1Optimizer = torch.optim.Adam(self.critic1.parameters(), lr=alpha)
		self.cr2Optimizer = torch.optim.Adam(self.critic2.parameters(), lr=alpha)

		self.mse_loss = nn.MSELoss(reduction='mean')

		self.free_exploration_step = free_exploration_step
		self.gamma = gamma
		self.alpha = alpha
		self.noise = noise

		self.tau = tau
		self.targetDeviation = targetDeviation
		self.clipWidth = clipWidth
		self.delay = delay

		self.train_step = 0
		self.current_step = 0

	def save_models(self, pathPrefix):
		torch.save(self.actor, pathPrefix +"_TD3_AC.pth")
		torch.save(self.critic1, pathPrefix + "_TD3_CR1.pth")
		torch.save(self.critic2, pathPrefix + "_TD3_CR2.pth")

	def load_models(self, pathPrefix):
		self.actor = torch.load(pathPrefix +"_TD3_AC.pth")
		self.critic1 = torch.load(pathPrefix + "_TD3_CR1.pth")
		self.critic2 = torch.load(pathPrefix + "_TD3_CR2.pth")

	def trainExpReplay(self, replayBuffer, batch_size):
		data = replayBuffer.sample(batch_size)

		# make smoothing noise
		smoothingNoise = []
		actDim = self.envDict.getActionDim()
		for j in range(batch_size):
			tmp = np.clip(np.random.normal(0, self.targetDeviation, actDim), -1 * self.clipWidth, self.clipWidth).tolist()
			smoothingNoise.append(tmp)

		# make train data
		states_cr = torch.FloatTensor(data[0])
		actions_cr = torch.FloatTensor(data[1])
		next_states_cr = torch.FloatTensor(data[2])
		rewards_cr = torch.FloatTensor(data[3])
		rewards_cr = torch.unsqueeze(rewards_cr, dim=1)
		tmp = []
		dones = data[4]
		for i in range(len(dones)):
			if dones[i]:
				tmp.append(0)
			else: tmp.append(self.gamma)
		dones_cr = torch.FloatTensor(tmp)
		dones_cr = torch.unsqueeze(dones_cr, dim=1)

		smoothingNoise_cr = torch.FloatTensor(smoothingNoise)
		#if self.envDict.getActionDim() == 1:
		#	smoothingNoise_cr = torch.unsqueeze(smoothingNoise_cr, dim=1)

		# calculate target value = delta_t(critic)
		smoothingAction_cr = torch.add(self.actor(next_states_cr), smoothingNoise_cr)
		smoothingAction_cr = torch.clamp(smoothingAction_cr, min=self.envDict.getActionRange(0)[0], max=self.envDict.getActionRange(0)[1])
		criticInput_ans_cr = torch.cat([next_states_cr, smoothingAction_cr], dim=1)
		ans1_nextStep = self.critic1(criticInput_ans_cr)
		ans2_nextStep = self.critic2(criticInput_ans_cr)
		ans_nextStep = torch.minimum(ans1_nextStep, ans2_nextStep)
		answers = torch.add(rewards_cr, torch.mul(dones_cr, ans_nextStep))

		#renewal(critic1)
		criticInputs1 = torch.cat([states_cr, actions_cr], dim=1)
		predictions1 = self.critic1(criticInputs1)
		cost1 = self.mse_loss(predictions1, answers.detach())
		self.cr1Optimizer.zero_grad()
		cost1.backward()
		self.cr1Optimizer.step()

		#renewal(critic2)
		criticInputs2 = torch.cat([states_cr, actions_cr], dim=1)
		predictions2 = self.critic2(criticInputs2)
		cost2 = self.mse_loss(predictions2, answers.detach())
		self.cr2Optimizer.zero_grad()
		cost2.backward()
		self.cr2Optimizer.step()

		if self.current_step == self.delay:
			#calculate target value(actor)
			states_ac = torch.FloatTensor(data[0])
			criticInputs = torch.cat([states_ac, self.actor(states_ac)], dim=1)
			criticResult = self.critic1(criticInputs)
			loss = torch.neg(torch.mean(criticResult))

			#renewal(actor)
			self.acOptimizer.zero_grad()
			loss.backward()
			self.acOptimizer.step()

			#clear current step buffer
			self.current_step = 0
		else:
			self.current_step = self.current_step + 1

class TD3AgentABL2(ACAgent):
	# without smoothing regularization

	def __init__(self, envName, free_exploration_step, gamma, alpha, noise, tau, targetDeviation, clipWidth, delay):
		self.actor = ActorNN(envName)
		self.critic1 = CriticNN(envName)
		self.critic2 = CriticNN(envName)

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.acOptimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
		self.cr1Optimizer = torch.optim.Adam(self.critic1.parameters(), lr=alpha)
		self.cr2Optimizer = torch.optim.Adam(self.critic2.parameters(), lr=alpha)

		self.targetActor = TargetActor(self.actor, tau, (self.envDict.getActionRange(0)[1] - self.envDict.getActionRange(0)[0]) / 2)
		self.targetCritic1 = TargetCritic(self.critic1, tau)
		self.targetCritic2 = TargetCritic(self.critic2, tau)

		self.mse_loss = nn.MSELoss(reduction='mean')

		self.free_exploration_step = free_exploration_step
		self.gamma = gamma
		self.alpha = alpha
		self.noise = noise

		self.tau = tau
		self.targetDeviation = targetDeviation
		self.clipWidth = clipWidth
		self.delay = delay

		self.train_step = 0
		self.current_step = 0

	def save_models(self, pathPrefix):
		torch.save(self.actor, pathPrefix +"_TD3_AC.pth")
		torch.save(self.critic1, pathPrefix + "_TD3_CR1.pth")
		torch.save(self.critic2, pathPrefix + "_TD3_CR2.pth")
		self.targetActor.saveParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic1.saveParameters(pathPrefix + "_TD3_TCR1")
		self.targetCritic2.saveParameters(pathPrefix + "_TD3_TCR2")

	def load_models(self, pathPrefix):
		self.actor = torch.load(pathPrefix +"_TD3_AC.pth")
		self.critic1 = torch.load(pathPrefix + "_TD3_CR1.pth")
		self.critic2 = torch.load(pathPrefix + "_TD3_CR2.pth")
		self.targetActor.loadParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic1.loadParameters(pathPrefix + "_TD3_TCR1")
		self.targetCritic2.loadParameters(pathPrefix + "_TD3_TCR2")

	def trainExpReplay(self, replayBuffer, batch_size):
		data = replayBuffer.sample(batch_size)

		# make train data
		states_cr = torch.FloatTensor(data[0])
		actions_cr = torch.FloatTensor(data[1])
		next_states_cr = torch.FloatTensor(data[2])
		rewards_cr = torch.FloatTensor(data[3])
		rewards_cr = torch.unsqueeze(rewards_cr, dim=1)
		tmp = []
		dones = data[4]
		for i in range(len(dones)):
			if dones[i]:
				tmp.append(0)
			else: tmp.append(self.gamma)
		dones_cr = torch.FloatTensor(tmp)
		dones_cr = torch.unsqueeze(dones_cr, dim=1)

		# calculate target value = delta_t(critic)
		criticInput_ans_cr = torch.cat([next_states_cr, self.targetActor.propagation(next_states_cr)], dim=1)
		ans1_nextStep = self.targetCritic1.propagation(criticInput_ans_cr)
		ans2_nextStep = self.targetCritic2.propagation(criticInput_ans_cr)
		ans_nextStep = torch.minimum(ans1_nextStep, ans2_nextStep)
		answers = torch.add(rewards_cr, torch.mul(dones_cr, ans_nextStep))

		#renewal(critic1)
		criticInputs1 = torch.cat([states_cr, actions_cr], dim=1)
		predictions1 = self.critic1(criticInputs1)
		cost1 = self.mse_loss(predictions1, answers.detach())
		self.cr1Optimizer.zero_grad()
		cost1.backward()
		self.cr1Optimizer.step()

		#renewal(critic2)
		criticInputs2 = torch.cat([states_cr, actions_cr], dim=1)
		predictions2 = self.critic2(criticInputs2)
		cost2 = self.mse_loss(predictions2, answers.detach())
		self.cr2Optimizer.zero_grad()
		cost2.backward()
		self.cr2Optimizer.step()

		if self.current_step == self.delay:
			#calculate target value(actor)
			states_ac = torch.FloatTensor(data[0])
			criticInputs = torch.cat([states_ac, self.actor(states_ac)], dim=1)
			criticResult = self.critic1(criticInputs)
			loss = torch.neg(torch.mean(criticResult))

			#renewal(actor)
			self.acOptimizer.zero_grad()
			loss.backward()
			self.acOptimizer.step()

			#renewal(targetActor, targetCritic)
			self.targetActor.renewal()
			self.targetCritic1.renewal()
			self.targetCritic2.renewal()

			#clear current step buffer
			self.current_step = 0
		else:
			self.current_step = self.current_step + 1

class TD3AgentABL3(ACAgent):
	# without delayed policy update

	def __init__(self, envName, free_exploration_step, gamma, alpha, noise, tau, targetDeviation, clipWidth, delay):
		self.actor = ActorNN(envName)
		self.critic1 = CriticNN(envName)
		self.critic2 = CriticNN(envName)

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.acOptimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
		self.cr1Optimizer = torch.optim.Adam(self.critic1.parameters(), lr=alpha)
		self.cr2Optimizer = torch.optim.Adam(self.critic1.parameters(), lr=alpha)

		self.targetActor = TargetActor(self.actor, tau, (self.envDict.getActionRange(0)[1] - self.envDict.getActionRange(0)[0]) / 2)
		self.targetCritic1 = TargetCritic(self.critic1, tau)
		self.targetCritic2 = TargetCritic(self.critic2, tau)

		self.mse_loss = nn.MSELoss(reduction='mean')

		self.free_exploration_step = free_exploration_step
		self.gamma = gamma
		self.alpha = alpha
		self.noise = noise

		self.tau = tau
		self.targetDeviation = targetDeviation
		self.clipWidth = clipWidth
		self.delay = delay

		self.train_step = 0

	def save_models(self, pathPrefix):
		torch.save(self.actor, pathPrefix +"_TD3_AC.pth")
		torch.save(self.critic1, pathPrefix + "_TD3_CR1.pth")
		torch.save(self.critic2, pathPrefix + "_TD3_CR2.pth")
		self.targetActor.saveParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic1.saveParameters(pathPrefix + "_TD3_TCR1")
		self.targetCritic2.saveParameters(pathPrefix + "_TD3_TCR2")

	def load_models(self, pathPrefix):
		self.actor = torch.load(pathPrefix +"_TD3_AC.pth")
		self.critic1 = torch.load(pathPrefix + "_TD3_CR1.pth")
		self.critic2 = torch.load(pathPrefix + "_TD3_CR2.pth")
		self.targetActor.loadParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic1.loadParameters(pathPrefix + "_TD3_TCR1")
		self.targetCritic2.loadParameters(pathPrefix + "_TD3_TCR2")

	def trainExpReplay(self, replayBuffer, batch_size):
		data = replayBuffer.sample(batch_size)

		smoothingNoise = []
		actDim = self.envDict.getActionDim()
		for j in range(batch_size):
			tmp = np.clip(np.random.normal(0, self.targetDeviation, actDim), -1 * self.clipWidth, self.clipWidth).tolist()
			smoothingNoise.append(tmp)

		# make train data
		states_cr = torch.FloatTensor(data[0])
		actions_cr = torch.FloatTensor(data[1])
		next_states_cr = torch.FloatTensor(data[2])
		rewards_cr = torch.FloatTensor(data[3])
		rewards_cr = torch.unsqueeze(rewards_cr, dim=1)
		tmp = []
		dones = data[4]
		for i in range(len(dones)):
			if dones[i]:
				tmp.append(0)
			else: tmp.append(self.gamma)
		dones_cr = torch.FloatTensor(tmp)
		dones_cr = torch.unsqueeze(dones_cr, dim=1)

		smoothingNoise_cr = torch.FloatTensor(smoothingNoise)
		#if self.envDict.getActionDim() == 1:
		#	smoothingNoise_cr = torch.unsqueeze(smoothingNoise_cr, dim=1)

		# calculate target value = delta_t(critic)
		smoothingAction_cr = torch.add(self.targetActor.propagation(next_states_cr), smoothingNoise_cr)
		smoothingAction_cr = torch.clamp(smoothingAction_cr, min=self.envDict.getActionRange(0)[0], max=self.envDict.getActionRange(0)[1])
		criticInput_ans_cr = torch.cat([next_states_cr, smoothingAction_cr], dim=1)
		ans1_nextStep = self.targetCritic1.propagation(criticInput_ans_cr)
		ans2_nextStep = self.targetCritic2.propagation(criticInput_ans_cr)
		ans_nextStep = torch.minimum(ans1_nextStep, ans2_nextStep)
		answers = torch.add(rewards_cr, torch.mul(dones_cr, ans_nextStep))

		#renewal(critic1)
		criticInputs1 = torch.cat([states_cr, actions_cr], dim=1)
		predictions1 = self.critic1(criticInputs1)
		cost1 = self.mse_loss(predictions1, answers.detach())
		self.cr1Optimizer.zero_grad()
		cost1.backward()
		self.cr1Optimizer.step()

		#renewal(critic2)
		criticInputs2 = torch.cat([states_cr, actions_cr], dim=1)
		predictions2 = self.critic2(criticInputs2)
		cost2 = self.mse_loss(predictions2, answers.detach())
		self.cr2Optimizer.zero_grad()
		cost2.backward()
		self.cr2Optimizer.step()

		#calculate target value(actor)
		states_ac = torch.FloatTensor(data[0])
		criticInputs = torch.cat([states_ac, self.actor(states_ac)], dim=1)
		criticResult = self.critic1(criticInputs)
		loss = torch.neg(torch.mean(criticResult))

		#renewal(actor)
		self.acOptimizer.zero_grad()
		loss.backward()
		self.acOptimizer.step()

		#renewal(targetActor, targetCritic)
		self.targetActor.renewal()
		self.targetCritic1.renewal()
		self.targetCritic2.renewal()

class TD3AgentABL4(ACAgent):

	def __init__(self, envName, free_exploration_step, gamma, alpha, noise, tau, targetDeviation, clipWidth, delay):
		self.actor = ActorNN(envName)
		self.critic = CriticNN(envName)

		self.envDict = gbm.EnvDict()
		self.envDict.setEnv(envName)

		self.acOptimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
		self.crOptimizer = torch.optim.Adam(self.critic.parameters(), lr=alpha)

		self.targetActor = TargetActor(self.actor, tau, (self.envDict.getActionRange(0)[1] - self.envDict.getActionRange(0)[0]) / 2)
		self.targetCritic = TargetCritic(self.critic, tau)

		self.mse_loss = nn.MSELoss(reduction='mean')

		self.free_exploration_step = free_exploration_step
		self.gamma = gamma
		self.alpha = alpha
		self.noise = noise

		self.tau = tau
		self.targetDeviation = targetDeviation
		self.clipWidth = clipWidth
		self.delay = delay

		self.train_step = 0
		self.current_step = 0

	def save_models(self, pathPrefix):
		torch.save(self.actor, pathPrefix +"_TD3_AC.pth")
		torch.save(self.critic, pathPrefix + "_TD3_CR.pth")
		self.targetActor.saveParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic.saveParameters(pathPrefix + "_TD3_TCR")

	def load_models(self, pathPrefix):
		self.actor = torch.load(pathPrefix +"_TD3_AC.pth")
		self.critic = torch.load(pathPrefix + "_TD3_CR.pth")
		self.targetActor.loadParameters(pathPrefix + "_TD3_TAC")
		self.targetCritic.loadParameters(pathPrefix + "_TD3_TCR")

	def trainExpReplay(self, replayBuffer, batch_size):
		data = replayBuffer.sample(batch_size)

		smoothingNoise = []
		actDim = self.envDict.getActionDim()
		for j in range(batch_size):
			tmp = np.clip(np.random.normal(0, self.targetDeviation, actDim), -1 * self.clipWidth, self.clipWidth).tolist()
			smoothingNoise.append(tmp)

		# make train data
		states_cr = torch.FloatTensor(data[0])
		actions_cr = torch.FloatTensor(data[1])
		next_states_cr = torch.FloatTensor(data[2])
		rewards_cr = torch.FloatTensor(data[3])
		rewards_cr = torch.unsqueeze(rewards_cr, dim=1)
		tmp = []
		dones = data[4]
		for i in range(len(dones)):
			if dones[i]:
				tmp.append(0)
			else: tmp.append(self.gamma)
		dones_cr = torch.FloatTensor(tmp)
		dones_cr = torch.unsqueeze(dones_cr, dim=1)

		smoothingNoise_cr = torch.FloatTensor(smoothingNoise)
		#if self.envDict.getActionDim() == 1:
		#	smoothingNoise_cr = torch.unsqueeze(smoothingNoise_cr, dim=1)

		# calculate target value = delta_t(critic)
		smoothingAction_cr = torch.add(self.targetActor.propagation(next_states_cr), smoothingNoise_cr)
		smoothingAction_cr = torch.clamp(smoothingAction_cr, min=self.envDict.getActionRange(0)[0], max=self.envDict.getActionRange(0)[1])
		criticInput_ans_cr = torch.cat([next_states_cr, smoothingAction_cr], dim=1)
		ans_nextStep = self.targetCritic.propagation(criticInput_ans_cr)
		answers = torch.add(rewards_cr, torch.mul(dones_cr, ans_nextStep))

		#renewal(critic)
		criticInputs = torch.cat([states_cr, actions_cr], dim=1)
		predictions = self.critic(criticInputs)
		cost = self.mse_loss(predictions, answers.detach())
		self.crOptimizer.zero_grad()
		cost.backward()
		self.crOptimizer.step()

		if self.current_step == self.delay:
			#calculate target value(actor)
			states_ac = torch.FloatTensor(data[0])
			criticInputs = torch.cat([states_ac, self.actor(states_ac)], dim=1)
			criticResult = self.critic(criticInputs)
			loss = torch.neg(torch.mean(criticResult))

			#renewal(actor)
			self.acOptimizer.zero_grad()
			loss.backward()
			self.acOptimizer.step()

			#renewal(targetActor, targetCritic)
			self.targetActor.renewal()
			self.targetCritic.renewal()

			#clear current step buffer
			self.current_step = 0
		else:
			self.current_step = self.current_step + 1