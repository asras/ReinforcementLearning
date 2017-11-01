#Get observation
#Get action from network
#Do env.step
#Save observation in memory for experience replay
#Do learning (against frozen Q-clone)


import gym
import numpy as np
import tensorflow as tf
import itertools
from TFNNs import SimpleQLearner

number_of_episode = 2

env = gym.make("Go9x9-v0")

learner = SimpleQLearner()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i_episode in range(number_of_episode):
    print("Beginning episode: {}".format(i_episode))
    previous_observation = env.reset()
    for t in itertools.count():
        print("Timestep {}  ".format(t), end="\r")
        policy = learner.epsilon_greedy_policy(sess, np.reshape(previous_observation, [1, 243]), 0.1)
        action = np.random.choice(len(policy), p=policy)
        observation, reward, done, _ = env.step(action)
        
        target_value = reward + np.max(learner.predict(sess, np.reshape(observation, [1,243])))
        learner.update(sess, np.reshape(previous_observation, [1,243]), np.reshape(action, [1]), np.reshape(target_value, [1]))

        previous_observation = observation
        if done or t > 20:
            break
        















