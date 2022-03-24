import numpy as np
import torch
import gym
from Utilities import *

class DQL:
    """ Parameters:
            - replay_buffer : enables DQN with experience replay buffer
            - er_size : experience replay buffer size
            - episode_size : size of the episode to consider
            - timesteps_size : numer of timesteps per episode to consider
            - model : Deep Learning agent (model in pytorch)
            - env : Environment to train our model 
    """
    def __init__(self,  replay_buffer=True,
                        er_size = None,
                        episode_size = None,
                        timesteps_size = None,
                        policy = None,
                        epsilon = None,
                        temp = None,
                        model = None,
                        env = None,
                ):
        self.experiece_replay = replay_buffer
        self.er_size = er_size
        self.episode_size = episode_size
        self.timestep_size = timesteps_size
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        self.model = model
        self.env = env
              

    def __call__(self):

        env = gym.make(self.env)
        env.reset()
        episode_sequence = []
        function_values = []

        # Iterate over episodes
        for ep in range(self.episode_size):
            # Initialize sequence s1 = {x1} and preprocess f1 = f(s1)
            s1 = env.render(mode='rgb_array')
            episode_sequence.append(s1)
            obs, rew, done, info = env.step(env.action_space.sample())
            function_values.append(sigmoid(self.model.forward(obs)))

            # Iterate over timesteps
            for t in range(self.timestep_size):
                pass


        self.env.close()


    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # Randomly generate a value between [0,1] with a uniform distribution
            p = np.random.uniform(0,1,1)[0]
            if p < epsilon:
                # Select random action
                a = np.random.randint(0,self.n_actions)
            else:
                # Select most probable action
                a = argmax(self.Q_sa[s])  
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # we use the provided softmax function in Helper.py
            probs = softmax(self.Q_sa[s], temp)
            a = np.random.choice(range(0, self.n_actions),p=probs)
        return a




    