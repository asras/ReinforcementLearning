import gym
import time
import numpy as np
import matplotlib.pyplot as plt

def Learn(currentPolicy, previousPolicy, currentReward, previousBestReward):
	if (currentReward > previousBestReward):
		return list(currentPolicy), currentReward, True
	return list(previousPolicy), previousBestReward, False


def Act(state, policy):
	preaction = 0
	for j in range(4):
		preaction += policy[j]*state[j]

	action = int(round(1/(1+np.exp(-preaction))))
	return action


def main():
	env = gym.make('MountainCar-v0')

	## Save/Remember: current action, previous action, current observation, previous observation
	rewardsforplotting = []

	policy = [np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1), np.random.uniform(-1,1)]


	bestRewardSoFar = 0
	currentBestPolicy = policy
	for i_episode in range(1000):
		observation = env.reset()


		
		accumulatedReward = 0
		currentTrialPolicy = list(currentBestPolicy) #We have to make a copy here. list() does this.
		#Vary policy somehow
		scalefactor = np.random.uniform(0.8, 1.2)
		randindex = np.random.randint(4)
		signchange = int((round(np.random.random())-0.5)*2)
		currentTrialPolicy[randindex] *= scalefactor*signchange
		for ack in range(10):
			observation = env.reset()
			for t in range(200):
				##env.render()
				action = Act(observation, currentTrialPolicy)
				observation, reward, done, info = env.step(action)
				accumulatedReward += reward
				
				#time.sleep(0.5)
				if done:
					#print("Episode finished after {} timesteps".format(t+1))
					break
				
		currentBestPolicy, bestRewardSoFar, updated = Learn(currentTrialPolicy, currentBestPolicy, accumulatedReward, bestRewardSoFar)
		
		
		#print(policy)
		# updated = False
		# for j in range(len(policy)):
		# 	if (abs(currentBestPolicy[j]-currentTrialPolicy[j])<10**-5):
		# 		updated = True
		if (updated):
			rewardsforplotting.append(accumulatedReward)
		#print("Updated?", updated)




	plt.plot(rewardsforplotting)
	plt.xlabel("Policy number")
	plt.ylabel("Expected reward for 10 rounds")
	plt.show()

	##Render the best policy
	while(True):
		observation = env.reset()
		for t in range(200):
			env.render()
			action = Act(observation, currentBestPolicy)
			observation, reward, done, info = env.step(action)
			if done:
				break;
		print("Finished after " + str(t+1) + " timesteps.")
		runagain = input("Run again? Y/N:")
		if (runagain.lower() != "y"):
			break


	#Data: currentBestPolicy, currentTrialPolicy, currentReward, currentBestReward
	# currentTrialPolicy <- Vary(currentBestPolicy)
	# Run environment: currentReward += observed reward at each timestep
	# currentBestPolicy <- currentReward > currentBestReward ? currentTrialPolicy : currentBestPolicy
	# currentBestReward = max(currentReward, currentBestReward)

if __name__ == "__main__":
	main()
	