import numpy as np
import copy



class QLearner:

    def __init__(self, learning_rate = 0.1, replay_number = 10
    , learn_switch_count=20):
        self.memory = []
        self.network = self.create_network()
        self.network_copy = self.copy_network()
        self.replay_number = replay_number
        self.learn_count = 0
        self.learn_switch_count = learn_switch_count


    def create_network(self):
        raise NotImplementedError

    def copy_network(self):
        raise NotImplementedError

    def get_action(self, observation):
        raise NotImplementedError
    
    def get_epsilon_greedy_action(self, observation):
        raise NotImplementedError

    
    def store_in_memory(self, data):
        self.memory.append(data)
    

    def learn(self):
        self.learn_count += 1
        if (self.learn_count >= self.learn_switch_count):
            self.network_copy = self.copy_network()
            self.learn_count = 0
    
        ##Get replays (st, at, rt, st+1)
        n_replays = np.min([self.replay_number, len(self.memory)])
        replay_data = np.random.choice(self.memory, n_replays)
        ##for each:
        ###Calculate gradient  Q(st, at)
        ###Update with respect target (rt + max_a(Qst+1, a))Grad(Q(st, at))


        raise NotImplementedError

    





        #action = learner.get_action(observation)
        #observation, reward, done, _ = env.step(action)
        #data = np.array([previous_observation, action, reward])
        #learner.store_in_memory(data)
        #learner.learn()


















