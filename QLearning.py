#Get observation
#Get action from network
#Do env.step
#Save observation in memory for experience replay
#Do learning (against frozen Q-clone)


import gym
import numpy as np
from QLearner import QLearner

number_of_episode = 100

env = gym.make("SpaceInvaders-v0")

learner = QLearner()


for i_episode in range(number_of_episode):
    previous_observation = env.reset()
    while True:
        action = learner.get_epsilon_greedy_action(observation)
        observation, reward, done, _ = env.step(action)
        data = np.array([previous_observation, action, reward])
        learner.store_in_memory(data)
        learner.learn()


        previous_observation = observation
        if done:
            break
        















