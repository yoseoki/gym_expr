from tqdm import tqdm
import gym
import pybullet_envs
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 1. constructor : no parameters
# 2. setEnv : 
#     param : envName(type of the environment, 'P' or 'B' or 'H')
#     return : None
#     set environment type
# 3. getActionDim : 
#     param : None
#     return : dimension of environment's action space(int)
#     get Dimension of the action space
# 4. getStateDim : 
#     param : None
#     return : dimension of environment's state space(int)
#     get Dimension of the state space
# 5. getActionRange : 
#     param : dimNum(number of the axis)
#     return : interval of the axis(tuple of int)
#     get interval of specific axis
# 6. getStateRange : 
#     param : dimNum(number of the axis)
#     return : interval of the axis(tuple of int)
#     get interval of specific axis
class EnvDict:

	def __init__(self):
		self.envName = ""
		self.actionDim = 0
		self.stateDim = 0
		self.actionRange = None
		self.stateRange = None

	def setEnv(self, envName):
		self.envName = envName
		if 'P' in envName:
			self.actionDim = 1
			self.actionRange = [(-2, 2)]
			self.stateDim = 3
			self.stateRange = [(-1, 1), (-1, 1), (-8, 8)]
		elif 'B' in envName:
			self.actionDim = 4
			self.actionRange = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
			self.stateDim = 24
			self.stateRange = []
			for i in range(24):
				self.stateRange.append((-float('inf'), float('inf')))
		elif 'H' in envName:
			self.actionDim = 17
			self.actionRange = []
			for i in range(17):
				self.actionRange.append((-1, 1))
			self.stateDim = 44
			self.stateRange = []
			for i in range(44):
				self.stateRange.append((-float('inf'), float('inf')))
		else:
			print("There's no environment which has name start with {}!".format(envName))

	def getEnvName(self):
		return self.envName

	def getActionDim(self):
		return self.actionDim

	def getStateDim(self):
		return self.stateDim

	def getActionRange(self, dimNum=-1):
		if dimNum == -1:
			return self.actionRange
		else:
			return self.actionRange[dimNum]

	def getStateRange(self, dimNum=-1):
		if dimNum == -1:
			return self.stateRange
		else:
			return self.stateRange[dimNum]

class ReplayBuffer:

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.buffer = []

	def getBufferSize(self):
		return len(self.buffer)

	def bufferClear(self):
		self.buffer.clear()

	def add(self , state , action , next_state , reward , done):
		data = [state, action, next_state, reward, done]
		if len(self.buffer) < self.buffer_size:
			self.buffer.append(data)
		else: # buffer is full(len(self.buffer) >= buffer_size)
			self.buffer.pop(0)
			self.buffer.append(data)

	def sample(self , batch_size):
		result = [[], [], [], [], []]
		for i in range(batch_size):
			data = self.buffer[rd.randrange(0, len(self.buffer))]
			for j in range(5):
				result[j].append(data[j])
		return result

class Agent:
	def save_models(self, path):
		pass
	def load_models(self, path):
		pass
	def select_action(self, state):
		pass
	def select_exploratory_action(self, state):
		pass
	def train(self, state, action, next_state, reward, done):
		pass

class SampleAgent(Agent):

	def __init__(self, envName):
		self.envDict = EnvDict(envName)

	def select_action(self, state):
		result = []
		for i in range(self.envDict.getActionDim()):
			result.append(0.0)
		return result

	def select_exploratory_action(self, state):
		result = []
		actLen = self.envDict.getActionRange(0)[1] - self.envDict.getActionRange(0)[0]
		for i in range(self.envDict.getActionDim()):
			action = rp.random.rand() - 0.5
			action = action * actLen
			result.append(action)
		return result

class Trainer:

	def __init__(self, envName, agent, max_step):
		self.max_step = max_step
		self.agent = agent
		self.envDict = EnvDict()
		self.envDict.setEnv(envName)

	def train(self, trainEnv, trainSeed, testSeedContainer, frequencyNum, testNum, savePrefix="tmp/", plotMode="QM", trainRenderFlag=False, saveFlag=False, plotFlag=False):
		trainEnv.seed(trainSeed)

		state = trainEnv.reset()

		resultContainer = []
		ticksContainer = []

		for t in tqdm(range(self.max_step)):
			if trainRenderFlag:
				trainEnv.render()
			action = self.agent.select_exploratory_action(state)
			next_state, reward, done, info = trainEnv.step(action)
			self.agent.train(state, action, next_state, reward, done)
			state = next_state

			# test(while training)
			if t % frequencyNum == 0:
				tmp = []
				for testSeed in testSeedContainer:
					testEnv = None
					if "P" in self.envDict.getEnvName():
						testEnv = gym.make('Pendulum-v0')
					elif "B" in self.envDict.getEnvName():
						testEnv = gym.make('BipedalWalker-v3')
					elif "H" in self.envDict.getEnvName():
						testEnv = gym.make('HumanoidBulletEnv-v0')

					result = evalAgent_cumulative(testEnv, self.agent, testSeed, testNum)
					tmp.extend(result)
					testEnv.close()

				resultContainer.append(tmp)

				ticksContainer.append(int(t / frequencyNum) + 1)
				
				if saveFlag:
					self.agent.save_models(savePrefix + "(tmp" + str(int(t / frequencyNum) + 1).zfill(3) + ")")
				print("")
				print("average of cumulative rewards: {}".format(sum(tmp) / len(tmp)))

			if done:
				state = trainEnv.reset()

		if saveFlag:
			self.saveResult(resultContainer, savePrefix, testNum * len(testSeedContainer))
		if plotFlag:
			self.plotResult(resultContainer, ticksContainer, savePrefix, frequencyNum, testNum * len(testSeedContainer))

	def saveResult(self, resultContainer, savePrefix, dataNum, trainFlag=True):

		# save model(after training)
		if trainFlag:
			self.agent.save_models(savePrefix + " (final)")

		# save result container
		f = open(savePrefix +  "result_container.txt", 'w')
		for i in range(len(resultContainer)):
			for j in range(dataNum):
				f.write(str(resultContainer[i][j]))
				if j != (dataNum - 1):
					f.write(",")
			if i != len(resultContainer) - 1:
				f.write("\n")
		f.close()

		# calculate result mean
		resultMeanContainer = []
		for i in range(len(resultContainer)):
			resultMeanContainer.append(sum(resultContainer[i]) / len(resultContainer[i]))

		# save result(mean) container
		f = open(savePrefix +  "result_mean_container.txt", 'w')
		for i in range(len(resultMeanContainer)):
			f.write(str(resultMeanContainer[i]))
			if i != len(resultMeanContainer) - 1:
				f.write(",")
		f.write("\n")
		f.close()

	def plotResult(self, resultContainer, ticksContainer, savePrefix, frequencyNum, dataNum, plotMode="QM"):
		# calculate result mean
		resultMeanContainer = []
		for i in range(len(resultContainer)):
			resultMeanContainer.append(sum(resultContainer[i]) / len(resultContainer[i]))
		
		# plot cumulative rewards
		if "B" in plotMode:
			plt.boxplot(resultContainer)
			plt.xticks(ticksContainer, ticksContainer)
			plt.xlabel('period ({} step)'.format(frequencyNum))
			plt.ylabel('cumulative reward')
			plt.savefig(savePrefix + "graph_box.png")
			if ("Q" in plotMode) or ("M" in plotMode):
				plt.clf()
		if "Q" in plotMode:
			resultQ1 = []
			resultQ2 = []
			resultQ3 = []
			idxQ1 = dataNum * 0.25
			idxQ2 = dataNum * 0.5
			idxQ3 = dataNum * 0.75
			for result in resultContainer:
				result.sort()
				resultQ1.append(result[int(idxQ1)])
				resultQ2.append(result[int(idxQ2)])
				resultQ3.append(result[int(idxQ3)])
			plt.plot(ticksContainer, resultQ1)
			plt.plot(ticksContainer, resultQ3)
			plt.fill_between(ticksContainer, resultQ1, resultQ3)
			plt.plot(ticksContainer, resultQ2)
			plt.xlabel('period ({} step)'.format(frequencyNum))
			plt.ylabel('cumulative reward')
			plt.savefig(savePrefix + "graph_quantile.png")
			if "M" in plotMode:
				plt.clf()
		if "M" in plotMode:
			plt.plot(ticksContainer, resultMeanContainer)
			plt.xlabel('period ({} step)'.format(frequencyNum))
			plt.ylabel('cumulative reward')
			plt.savefig(savePrefix + "graph_mean.png")
			plt.clf()

	def test(self, testSeedContainer, testNum, savePrefix="tmp/", testRenderFlag=True, saveFlag=False, plotFlag=False):
		resultContainer = []

		for testSeed in testSeedContainer:
			testEnv = None
			if "P" in self.envDict.getEnvName():
				testEnv = gym.make('Pendulum-v0')
			elif "B" in self.envDict.getEnvName():
				testEnv = gym.make('BipedalWalker-v3')
			elif "H" in self.envDict.getEnvName():
				testEnv = gym.make('HumanoidBulletEnv-v0')

			print("testSeed : {}".format(testSeed))
			result = evalAgent_cumulative(testEnv, self.agent, testSeed, testNum, renderFlag=testRenderFlag)
			resultContainer.append(result)
			for i in range(testNum):
				print("{}th cumulative reward : {}".format(i, result[i]))
			testEnv.close()

		if saveFlag:
			self.saveResult(resultContainer, savePrefix, testNum, trainFlag=False)
		if plotFlag:
			ticksContainer = []
			for i in range(len(testSeedContainer)):
				ticksContainer.append(i + 1)
			self.plotResult(resultContainer, ticksContainer, savePrefix, 200, testNum)

class RBTrainer(Trainer):

	def __init__(self, envName, agent, max_step, batch_size):
		self.max_step = max_step
		self.batch_size = batch_size

		self.envDict = EnvDict()
		self.envDict.setEnv(envName)

		self.agent = agent
		self.replayBuffer = ReplayBuffer(max_step)

	def train(self, trainEnv, trainSeed, testSeedContainer, frequencyNum, testNum, savePrefix="tmp/", plotMode="QM", trainRenderFlag=False, saveFlag=False, plotFlag=False):
		trainEnv.seed(trainSeed)

		state = trainEnv.reset()

		resultContainer = []
		ticksContainer = []

		for t in tqdm(range(self.max_step)):
			if trainRenderFlag:
				trainEnv.render()
			action = self.agent.select_exploratory_action(state)
			next_state, reward, done, info = trainEnv.step(action)
			self.replayBuffer.add(state, action, next_state, reward, done)
			state = next_state
			if t >= self.batch_size:
				self.agent.trainExpReplay(self.replayBuffer, self.batch_size)

			# test(while training)
			if t % frequencyNum == 0:
				tmp = []
				for testSeed in testSeedContainer:
					testEnv = None
					if "P" in self.envDict.getEnvName():
						testEnv = gym.make('Pendulum-v0')
					elif "B" in self.envDict.getEnvName():
						testEnv = gym.make('BipedalWalker-v3')
					elif "H" in self.envDict.getEnvName():
						testEnv = gym.make('HumanoidBulletEnv-v0')

					result = evalAgent_cumulative(testEnv, self.agent, testSeed, testNum)
					tmp.extend(result)
					testEnv.close()

				resultContainer.append(tmp)
				ticksContainer.append(int(t / frequencyNum) + 1)
				if saveFlag:
					self.agent.save_models(savePrefix + "(tmp" + str(int(t / frequencyNum) + 1).zfill(3) + ")")
				print("")
				print("average of cumulative rewards: {}".format(sum(tmp) / len(tmp)))

			if done:
				state = trainEnv.reset()

		if saveFlag:
			self.saveResult(resultContainer, savePrefix, testNum * len(testSeedContainer))
		if plotFlag:
			self.plotResult(resultContainer, ticksContainer, savePrefix, frequencyNum, testNum * len(testSeedContainer))

def evalAgent_cumulative(env, agent, seed, evalCount, renderFlag=False, debugFlag=False):

	env.seed(seed)
	state = env.reset()

	evalCounter = 0
	evalContainer = []
	cumReward = 0

	while evalCounter < evalCount:
		if renderFlag:
			env.render()
		action = agent.select_action(state)
		if debugFlag:
			print("", end="\n1. Action : ")
			print(action)
		next_state, reward, done, info = env.step(action)
		cumReward = cumReward + reward
		state = next_state
		if done:
			state = env.reset()
			evalContainer.append(cumReward)
			cumReward = 0
			evalCounter = evalCounter + 1

	return evalContainer

def makeMetaFileQ(prefix, paramContainer):

	if not os.path.isdir("./" + prefix):
		os.mkdir("./" + prefix)
	f = open("./" + prefix + "/meta.txt", 'w')

	trainDate = datetime.today().strftime("%Y%m%d")

	if(len(paramContainer) == 15):
		f.write("train date : {}".format(trainDate))
		f.write("\n")
		f.write("train number : {}".format(paramContainer[0]))
		f.write("\n")
		f.write("train environment : {}".format(paramContainer[1]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("random seed of numpy : {}".format(paramContainer[2]))
		f.write("\n")
		f.write("random seed of random : {}".format(paramContainer[3]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("max step : {}".format(paramContainer[4]))
		f.write("\n")
		f.write("batch size : {}".format(paramContainer[5]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("train seed : {}".format(paramContainer[6]))
		f.write("\n")
		tmp = ""
		for i in range(len(paramContainer[7])):
			tmp = tmp + str(paramContainer[7][i])
			if i != len(paramContainer[7]) - 1:
				tmp = tmp + ", "
		f.write("test(while train) seed : " + tmp)
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("test has been executed every {} steps,".format(paramContainer[8]))
		f.write("\n")
		f.write("and record cumulative rewards of {} episodes".format(paramContainer[9]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("state space division size : {}".format(paramContainer[10]))
		f.write("\n")
		f.write("action space division size : {}".format(paramContainer[11]))
		f.write("\n")
		f.write("epsilon : {}".format(paramContainer[12]))
		f.write("\n")
		f.write("gamma : {}".format(paramContainer[13]))
		f.write("\n")
		f.write("alpha : {}".format(paramContainer[14]))
		f.write("\n")

		param = open("./" + prefix + "/param.txt", 'w')
		param.write(trainDate + str(paramContainer[0]))
		param.write("\n")
		param.write(str(paramContainer[1]))
		param.write("\n")
		param.write(str(paramContainer[10]))
		param.write("\n")
		param.write(str(paramContainer[11]))
		param.write("\n")
		param.write(str(paramContainer[12]))
		param.write("\n")
		param.write(str(paramContainer[13]))
		param.write("\n")
		param.write(str(paramContainer[14]))
		param.write("\n")
		param.close()
	else:
		f.write("test date : {}".format(datetime.today().strftime("%Y%m%d")))
		f.write("\n")
		f.write("test number : {}".format(paramContainer[0]))
		f.write("\n")
		f.write("test environment : {}".format(paramContainer[1]))
		f.write("\n")
		f.write("test model : {}".format(paramContainer[2]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("random seed of numpy : {}".format(paramContainer[3]))
		f.write("\n")
		f.write("random seed of random : {}".format(paramContainer[4]))
		f.write("\n")
		f.write("="*20)
		tmp = ""
		for i in range(len(paramContainer[5])):
			tmp = tmp + str(paramContainer[5][i])
			if i != len(paramContainer[5]) - 1:
				tmp = tmp + ", "
		f.write("test seed : " + tmp)
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("test record cumulative rewards of {} episodes".format(paramContainer[6]))
		f.write("\n")

	f.close()

def makeMetaFileAC(prefix, paramContainer):

	if not os.path.isdir("./" + prefix):
		os.mkdir("./" + prefix)
	f = open("./" + prefix + "/meta.txt", 'w')

	trainDate = datetime.today().strftime("%Y%m%d")

	if(len(paramContainer) == 15):
		f.write("train date : {}".format(trainDate))
		f.write("\n")
		f.write("train number : {}".format(paramContainer[0]))
		f.write("\n")
		f.write("train environment : {}".format(paramContainer[1]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("random seed of numpy : {}".format(paramContainer[2]))
		f.write("\n")
		f.write("random seed of random : {}".format(paramContainer[3]))
		f.write("\n")
		f.write("random seed of pytorch : {}".format(paramContainer[4]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("max step : {}".format(paramContainer[5]))
		f.write("\n")
		f.write("free exploration step : {}".format(paramContainer[6]))
		f.write("\n")
		f.write("batch size : {}".format(paramContainer[7]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("train seed : {}".format(paramContainer[8]))
		f.write("\n")
		tmp = ""
		for i in range(len(paramContainer[9])):
			tmp = tmp + str(paramContainer[9][i])
			if i != len(paramContainer[9]) - 1:
				tmp = tmp + ", "
		f.write("test(while train) seed : " + tmp)
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("test has been executed every {} steps,".format(paramContainer[10]))
		f.write("\n")
		f.write("and record cumulative rewards of {} episodes".format(paramContainer[11]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("gamma : {}".format(paramContainer[12]))
		f.write("\n")
		f.write("alpha : {}".format(paramContainer[13]))
		f.write("\n")
		f.write("noise : {}".format(paramContainer[14]))
		f.write("\n")

		param = open("./" + prefix + "/param.txt", 'w')
		param.write(trainDate + str(paramContainer[0]))
		param.write("\n")
		param.write(str(paramContainer[1]))
		param.write("\n")
		param.write(str(paramContainer[6]))
		param.write("\n")
		param.write(str(paramContainer[12]))
		param.write("\n")
		param.write(str(paramContainer[13]))
		param.write("\n")
		param.write(str(paramContainer[14]))
		param.write("\n")
		param.close()
	else:
		f.write("test date : {}".format(datetime.today().strftime("%Y%m%d")))
		f.write("\n")
		f.write("test number : {}".format(paramContainer[0]))
		f.write("\n")
		f.write("test environment : {}".format(paramContainer[1]))
		f.write("\n")
		f.write("test model : {}".format(paramContainer[2]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("random seed of numpy : {}".format(paramContainer[3]))
		f.write("\n")
		f.write("random seed of random : {}".format(paramContainer[4]))
		f.write("\n")
		f.write("random seed of pytorch : {}".format(paramContainer[5]))
		f.write("\n")
		f.write("="*20)
		tmp = ""
		for i in range(len(paramContainer[6])):
			tmp = tmp + str(paramContainer[6][i])
			if i != len(paramContainer[6]) - 1:
				tmp = tmp + ", "
		f.write("test seed : " + tmp)
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("test record cumulative rewards of {} episodes".format(paramContainer[7]))
		f.write("\n")

	f.close()

def makeMetaFileTD3(prefix, paramContainer):

	if not os.path.isdir("./" + prefix):
		os.mkdir("./" + prefix)
	f = open("./" + prefix + "/meta.txt", 'w')

	trainDate = datetime.today().strftime("%Y%m%d")

	if(len(paramContainer) == 19):
		f.write("train date : {}".format(trainDate))
		f.write("\n")
		f.write("train number : {}".format(paramContainer[0]))
		f.write("\n")
		f.write("train environment : {}".format(paramContainer[1]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("random seed of numpy : {}".format(paramContainer[2]))
		f.write("\n")
		f.write("random seed of random : {}".format(paramContainer[3]))
		f.write("\n")
		f.write("random seed of pytorch : {}".format(paramContainer[4]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("max step : {}".format(paramContainer[5]))
		f.write("\n")
		f.write("free exploration step : {}".format(paramContainer[6]))
		f.write("\n")
		f.write("batch size : {}".format(paramContainer[7]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("train seed : {}".format(paramContainer[8]))
		f.write("\n")
		tmp = ""
		for i in range(len(paramContainer[9])):
			tmp = tmp + str(paramContainer[9][i])
			if i != len(paramContainer[9]) - 1:
				tmp = tmp + ", "
		f.write("test(while train) seed : " + tmp)
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("test has been executed every {} steps,".format(paramContainer[10]))
		f.write("\n")
		f.write("and record cumulative rewards of {} episodes".format(paramContainer[11]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("gamma : {}".format(paramContainer[12]))
		f.write("\n")
		f.write("alpha : {}".format(paramContainer[13]))
		f.write("\n")
		f.write("noise : {}".format(paramContainer[14]))
		f.write("\n")
		f.write("tau : {}".format(paramContainer[15]))
		f.write("\n")
		f.write("target deviation : {}".format(paramContainer[16]))
		f.write("\n")
		f.write("width of clipping : {}".format(paramContainer[17]))
		f.write("\n")
		f.write("delay of actor training : {}".format(paramContainer[18]))
		f.write("\n")

		param = open("./" + prefix + "/param.txt", 'w')
		param.write(trainDate + str(paramContainer[0]))
		param.write("\n")
		param.write(str(paramContainer[1]))
		param.write("\n")
		param.write(str(paramContainer[6]))
		param.write("\n")
		param.write(str(paramContainer[12]))
		param.write("\n")
		param.write(str(paramContainer[13]))
		param.write("\n")
		param.write(str(paramContainer[14]))
		param.write("\n")
		param.write(str(paramContainer[15]))
		param.write("\n")
		param.write(str(paramContainer[16]))
		param.write("\n")
		param.write(str(paramContainer[17]))
		param.write("\n")
		param.write(str(paramContainer[18]))
		param.write("\n")
		param.close()
	else:
		f.write("test date : {}".format(datetime.today().strftime("%Y%m%d")))
		f.write("\n")
		f.write("test number : {}".format(paramContainer[0]))
		f.write("\n")
		f.write("test environment : {}".format(paramContainer[1]))
		f.write("\n")
		f.write("test model : {}".format(paramContainer[2]))
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("random seed of numpy : {}".format(paramContainer[3]))
		f.write("\n")
		f.write("random seed of random : {}".format(paramContainer[4]))
		f.write("\n")
		f.write("random seed of pytorch : {}".format(paramContainer[5]))
		f.write("\n")
		f.write("="*20)
		tmp = ""
		for i in range(len(paramContainer[6])):
			tmp = tmp + str(paramContainer[6][i])
			if i != len(paramContainer[6]) - 1:
				tmp = tmp + ", "
		f.write("test seed : " + tmp)
		f.write("\n")
		f.write("="*20)
		f.write("\n")
		f.write("test record cumulative rewards of {} episodes".format(paramContainer[7]))
		f.write("\n")

	f.close()