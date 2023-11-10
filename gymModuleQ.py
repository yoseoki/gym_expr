from tqdm import tqdm
import gym
import gymBasicModule as gbm
import random as rd
import numpy as np
import matplotlib.pyplot as plt

class TableQAgent(gbm.Agent):
	# action space : 1st-dimensional Real Vector Space of [-2, 2].
	# observation space(state space) : 3rd-dimensional Real Vector Space of [-1, 1] x [-1, 1] x [-8, 8].

	def __init__(self, stateDivisionSize, actionDivisionSize, epsilon, gamma, alpha):
		self.stateDivisionSize = stateDivisionSize
		self.actionDivisionSize = actionDivisionSize

		self.stateSize = stateDivisionSize**3 # env-dependent(Pendulum-v0)
		self.actionSize = actionDivisionSize # env-dependent(Pendulum-v0)

		self.qTable = []
		for k in range(self.stateSize):
			column = []
			for l in range(self.actionSize):
				column.append(np.random.normal(0.0, 1.0, 1)[0] * (0.1**8)) # initialize by (N(0,1) * (0.1)^8)
			self.qTable.append(column)

		self.epsilon = epsilon
		self.gamma = gamma
		self.alpha = alpha

		self.stateIntervalNum = 3 # env-dependent(Pendulum-v0)
		self.stateIntervalLength = [2.0, 2.0, 16.0] # env-dependent(Pendulum-v0)
		self.actionIntervalNum = 1 # env-dependent(Pendulum-v0)
		self.actionIntervalLength = [4.0] # env-dependent(Pendulum-v0)

	def save_models(self, pathPrefix):
		f = open(pathPrefix + "_QTable.txt", 'w')
		f.write(str(self.stateDivisionSize))
		f.write("\n")
		f.write(str(self.actionDivisionSize))
		f.write("\n")
		f.write(str(self.epsilon))
		f.write("\n")
		f.write(str(self.gamma))
		f.write("\n")
		f.write(str(self.alpha))
		f.write("\n")
		for i in range(self.stateSize):
			for j in range(self.actionSize):
				f.write(str(self.qTable[i][j]))
				if j != (self.actionSize -1):
					f.write(",")
			if i != (self.stateSize - 1):
				f.write("\n")
		f.close()

	def load_models(self, pathPrefix):
		self.qTable = []
		f = open(pathPrefix + "_QTable.txt", 'r')
		self.stateDivisionSize = int(f.readline())
		self.actionDivisionSize = int(f.readline())
		self.stateSize = self.stateDivisionSize**3
		self.actionSize = self.actionDivisionSize
		self.epsilon = float(f.readline())
		self.gamma = float(f.readline())
		self.alpha = float(f.readline())
		for i in range(self.stateSize):
			column = f.readline()
			self.qTable.append([])
			columnList = column.split(",")
			for j in range(self.actionSize):
				self.qTable[i].append(float(columnList[j]))
		f.close()


	def mappingState_(self, state):
		index = []
		result = 0
		for i in range(self.stateIntervalNum):
			increment = self.stateIntervalLength[i] / self.stateDivisionSize
			intervalMin = 0.0 - self.stateIntervalLength[i] / 2
			intervalMax = intervalMin + increment
			for j in range(self.stateDivisionSize):
				if intervalMin <= state[i] and state[i] <= intervalMax:
					index.append(j)
					break
				intervalMin = intervalMin + increment
				intervalMax = intervalMax + increment
		# if(len(index) != self.stateIntervalNum):
		
		for i in range(self.stateIntervalNum):
			result = result + (self.stateDivisionSize**(self.stateIntervalNum - 1 - i)) * index[i]
		return result

	def mappingAction_(self, action):
		index = []
		result = 0
		for i in range(self.actionIntervalNum):
			increment = self.actionIntervalLength[i] / self.actionDivisionSize
			intervalMin = 0.0 - self.actionIntervalLength[i] / 2
			intervalMax = intervalMin + increment
			for j in range(self.actionDivisionSize):
				if intervalMin <= action[i] and action[i] <= intervalMax:
					index.append(j)
					break
				intervalMin = intervalMin + increment
				intervalMax = intervalMax + increment
		# if(index == -1):

		for i in range(self.actionIntervalNum):
			result = result + (self.actionDivisionSize**(self.actionIntervalNum - 1 - i)) * index[i]
		return result

	def getAction_(self, actionIndex):
		increment = 4.0 / self.actionDivisionSize
		result = -2.0 + actionIndex * increment + 0.5 * increment
		return [result]

	def select_action(self, state):
		stateIndex = self.mappingState_(state)
		stateColumn = self.qTable[stateIndex]
		maxIndex = 0
		maxValue = stateColumn[0]
		for i in range(self.actionSize):
			if maxValue < stateColumn[i]:
				maxIndex = i
				maxValue = stateColumn[i]

		return self.getAction_(maxIndex)

	def select_exploratory_action(self, state):
		if rd.random() > self.epsilon: # by possiblity of (1- epsilon)
			return self.select_action(state)
		else: # by possiblity of (epsilon)
			return self.getAction_(rd.randrange(0, self.actionSize))

	def train(self, state, action, next_state, reward, done):
		oldValue = self.qTable[self.mappingState_(state)][self.mappingAction_(action)]
		newValue = reward
		if not done:
			potentialValue = max(self.qTable[self.mappingState_(next_state)])
			newValue = newValue + self.gamma * potentialValue

		self.qTable[self.mappingState_(state)][self.mappingAction_(action)] = oldValue + self.alpha * (newValue - oldValue)

	def trainExpReplay(self, replayBuffer, batch_size):
		if replayBuffer.getBufferSize() >= batch_size:
			data = replayBuffer.sample(batch_size)
			for i in range(batch_size):
				self.train(data[0][i], data[1][i], data[2][i], data[3][i], data[4][i])
		else: # buffer size is smaller than batch_size
			data = replayBuffer.buffer
			for i in range(len(data)):
				self.train(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4])