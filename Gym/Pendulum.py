##Implement "Random Learner" (copyright: Morten Madsen and Asbjørn Rasmussen) for Pendulum problem


import gym
import matplotlib.pyplot as plt 
import numpy as np



env = gym.make("Pendulum-v0") ##Check https://github.com/openai/gym/wiki/Pendulum-v0 for specs

numberOfEpisodes = 2000
numberOfTimesteps = 200
numberOfEvaluationSteps = 20

class Swinger(): ##Our Pendulum-env Reinforcement Learner
	
	def __init__(self):
		self.policy = [np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-10,10)]
		self.bestReward = -10**6
		self.observedRewards = []
		return

	def Act(self, observation, policyToUse=[]):
		if len(policyToUse) == 0:
			policyToUse = self.policy
		action = 0
		for j in range(len(policyToUse)):
			action += policyToUse[j]*observation[j]
		##See specs
		if (action <= -2.0):
			action = -1.99
		if (action >= 2.0):
			action = 1.99
		return action



	def Learn(self, trialPolicy, rewardForTrial):
		if (rewardForTrial > self.bestReward):
			self.bestReward = rewardForTrial
			self.policy = list(trialPolicy)
			self.observedRewards.append(rewardForTrial)
		return

	def GetTrialPolicy(self):
		trial = [np.random.uniform(-10,10) for _ in range(3)] #TODO find some way to extract size of obs space
		return trial




learner = Swinger()

for i_episode in range(numberOfEpisodes):

	#Get a trial policy
	trialPolicy = learner.GetTrialPolicy()
	accumReward = 0

	##Try out the policy for some number of steps
	for _ in range(numberOfEvaluationSteps):


		observation = env.reset()
		for t in range(numberOfTimesteps):
			action = learner.Act(observation, trialPolicy)
			observation, reward, done, info = env.step([action])
			accumReward += reward
			if done:
				break
	#See if policy is better than anything observed so far
	learner.Learn(trialPolicy, accumReward)


plt.plot(learner.observedRewards)
plt.xlabel("Policy number")
plt.ylabel("Accumulated reward")
plt.show()


while (True):
	observation = env.reset()
	for t in range(numberOfTimesteps):
		env.render()
		action = learner.Act(observation)
		observation, reward, done, info = env.step([action])
		if done:
			break
	runagain = input("Run again? Y/N: ")
	if (runagain.lower() != "y"):
		break