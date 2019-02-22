import os
import sys
import gym
import random
import collections
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F

EPISODES = 300

# reference
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
# 위 링크의 코드를 PyTorch 로 바꿈

class Agent(nn.Module):

	def __init__(self, nState, nAction):

		super(Agent, self).__init__()

		# 게임 플레이를 렌더링
		self.render = True

		# state 와 action 개수
		self.nState = nState
		self.nAction = nAction

		"""
		# DQN model, MLP
		self.model = nn.Sequential(
								nn.Linear(self.nState, 64),
								nn.ReLU(inplace = True),
								nn.Linear(64, 64),
								nn.ReLU(inplace = True),
								nn.Linear(64, 64),
								nn.ReLU(inplace = True),
								nn.Linear(64, self.nAction))
		self.model = self.model.to('cuda')
		self.model.train()

		self.modelTarget = nn.Sequential(
								nn.Linear(self.nState, 64),
								nn.ReLU(inplace = True),
								nn.Linear(64, 64),
								nn.ReLU(inplace = True),
								nn.Linear(64, 64),
								nn.ReLU(inplace = True),
								nn.Linear(64, self.nAction))
		self.modelTarget = self.modelTarget.to('cuda')
		self.modelTarget.eval()
		"""

		# DQN model, MLP
		self.model = nn.Sequential(
								nn.Linear(self.nState, 24),
								nn.ReLU(inplace = True),
								nn.Linear(24, 24),
								nn.ReLU(inplace = True),
								nn.Linear(24, self.nAction))
		self.model = self.model.to('cuda')
		self.model.train()

		self.modelTarget = nn.Sequential(
								nn.Linear(self.nState, 24),
								nn.ReLU(inplace = True),
								nn.Linear(24, 24),
								nn.ReLU(inplace = True),
								nn.Linear(24, self.nAction))
		self.updateTargetNetwork()
		self.modelTarget = self.modelTarget.to('cuda')
		self.modelTarget.eval()

		# DQN 의 parameter
		self.discount = 0.99
		self.learnigRate = 0.001
		self.epsilon = 1.0
		self.epsilonDecay = 0.999
		self.epsilonMin = 0.01
		self.batchSize = 64
		self.trainStart = 1000

		# 학습을 위한 optimizer 와 loss 정의
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr = self.learnigRate)

		# replay memory
		self.replayMemory = collections.deque(maxlen = 2000)

	# target network 의 parameter 는 주기적으로 복사하는 방식으로 업데이트
	def updateTargetNetwork(self):

		self.modelTarget.load_state_dict(self.model.state_dict())

	# epsilon greedy 방식을 사용
	# epsilon 의 확률로 임의의 action 을 취함 -> agent 가 다양한 경험을 하게 해줌
	def getAction(self, state):

		if np.random.random() < self.epsilon:

			return random.randrange(self.nAction)

		else:

			q = self.model(state).to('cpu')

			return torch.argmax(q[0]).item()

	# 주어진 sample 을 replay memory 에 저장
	def saveSample(self, state, action, reward, nextState, done):

		self.replayMemory.append((state, action, reward, nextState, done))

	def train(self):

		# epsilon 을 점차 감소하게 만듦
		if self.epsilon > self.epsilonMin:

			self.epsilon = self.epsilon * self.epsilonDecay

		# replay memory 에서 임의의 sample 을 추출
		trainData = random.sample(self.replayMemory, self.batchSize)

		# sample 에서 각각의 항목을 가져옴
		states = torch.zeros((self.batchSize, self.nState))
		nextStates = torch.zeros((self.batchSize, self.nState))
		
		actions = [e[1] for e in trainData]
		rewards = [e[2] for e in trainData]
		dones = [e[4] for e in trainData]

		for idx in range(self.batchSize):

			states[idx] = torch.as_tensor(trainData[idx][0])
			nextStates[idx] = torch.as_tensor(trainData[idx][3])

		states = states.to('cuda')
		nextStates = nextStates.to('cuda')

		# 현재 state 에 대한 model 의 q 
		# 다음 state 에 대한 target model 의 q
		q = self.model(states)
		qTarget = self.modelTarget(nextStates)

		# 벨만 최적 방정식을 이용한 업데이트 타깃
		for idx in range(self.batchSize):

			if dones[idx]:

				q[idx][actions[idx]] = rewards[idx]

			else:

				q[idx][actions[idx]] = rewards[idx] + self.discount * (torch.max(qTarget[idx]))

		loss = self.criterion(qTarget, q)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

if __name__ == "__main__":

	# CartPole-v1 환경, 한 episode 의 최대 길이는 500
	# 목적은 오래 버티기 -> 500 까지 버티면 가장 높은 reward
	env = gym.make('CartPole-v1')
	nState = env.observation_space.shape[0]
	nAction = env.action_space.n

	# DQN 에이전트 생성
	agent = Agent(nState, nAction)

	# episode 의 정보를 기록
	scores = list()
	episodes = list()

	for e in range(EPISODES):

		# done : 이번 episode 가 끝났는지를 나타내는 변수
		# score : reward 의 누적 값
		done = False
		score = 0

		# env 초기화
		state = env.reset()
		state = np.reshape(state, [1, nState])
		state = torch.from_numpy(state).type(torch.FloatTensor).to('cuda')

		while not done:

			# play 화면을 보여줌
			if agent.render:

				env.render()

			# 현재 state 에 기반해 action 을 선택
			# epsilon greedy 를 사용하기 때문에, 최선의 action 이 아닐 수도 있음
			action = agent.getAction(state)
			# environment 에서 action 을 취함
			nextState, reward, done, info = env.step(action)
			nextState = np.reshape(nextState, [1, nState])
			nextState = torch.from_numpy(nextState).type(torch.FloatTensor).to('cuda')
			# 오래 버티기를 실패한 경우엔 -100 점을 줌
			reward = reward if not done or score == 499 else -100

			# replay memory 에 현재 상황 (sample) 을 저장
			agent.saveSample(state, action, reward, nextState, done)
			
			# replay memory 에 담긴 sample 의 수가 미리 정한 것보다 많으면 학습
			if len(agent.replayMemory) >= agent.trainStart:
				
				agent.train()

			score = score + reward
			state = nextState

			if done:
				# episode 가 한 번 끝나면 target network 를 update 함
				agent.updateTargetNetwork()

				score = score if score == 500 else score + 100
				# episode 마다 학습 결과 출력
				scores.append(score)
				episodes.append(e)
				print('Episode : ', e)
				print('Score : ', score)
				print('Memory Length : ', len(agent.replayMemory))
				print('Epsilon : ', agent.epsilon)

				# 이전 10 개 epsiode 의 평균 점수가 490 이상이면 학습을 종료
				if np.mean(scores[-min(10, len(scores)):]) >= 490:

					plt.plot(episodes, scores)
					plt.show()

					env.close()
					sys.exit(0)
