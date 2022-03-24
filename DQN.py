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
            - epsilon : probability to choose a random action
            - temp : temperature parameter for softmax selection
            - model : Deep Learning agent (model in pytorch)
            - env : Environment to train our model 

        About the model:
        The model should output two parameters which indicates the probability
        for each action. The cart problem can be solved with the sigmoid
        and only one output on the model however, we will do it this way 
        in order to handle different environments with more actions later on.
    """
    def __init__(self,  replay_buffer=True,
                        er_size = None,
                        episode_size = None,
                        timesteps_size = None,
                        minibatch_size = None,
                        policy = None,
                        epsilon = None,
                        temp = None,
                        model = None,
                        input_is_img = None,
                        env = None,
                ):
        self.experiece_replay = replay_buffer
        self.er_size = er_size
        self.episode_size = episode_size
        self.timestep_size = timesteps_size
        self.minibatch_size = minibatch_size
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        self.model = model
        self.input_is_img = input_is_img
        self.env = env
              

    def __call__(self):

        # Initialize environment
        env = gym.make(self.env)
        initial_observation = env.reset()

        # Create data dictionary
        self.D = {   
                'episode_sequence': np.array(),
                'action_sequence' : np.array(),
                'reward_sequence' : np.array(),
                'done_sequence' : np.array(),
                'function_sequence' : np.array()
            }

        # TODO? Initialize action-value function Q with random weights?
        # Does it mean to just initialie the model? 
        # This has already done when passing the model above probably,
        # since weights get initialized

        # Iterate over episodes
        for ep in range(self.episode_size):
            # Initialize sequence s1 = {x1} and preprocess f1 = f(s1)

            # Control if we are using images as input of the model instead of observations
            if self.input_is_img:
                # TODO: add normalization. 
                # (also flattening is needed for DNN)
                # but it can be done in the model when creating it which is better
                s1 = env.render(mode='rgb_array')
            else:
                s1 = initial_observation
            self.D['episode_sequence'].append(s1)
            self.D['function_sequence'].append((self.model.forward(torch.FloatTensor(obs)))

            # Iterate over timesteps
            for t in range(self.timestep_size):
                # Select random action with probability epsilon or follow egreedy policy
                a = self.select_action(self.D['episode_sequence'][t],self.policy, self.epsilon, self.temp)
                
                # Execute action a_t in emulator and observe reward rt and image x_t+1
                obs, rew, done, _ = env.step(a)

                # Save all relevant data in self.D
                if self.input_is_img:
                    self.D['episode_sequence'].append(env.render(mode='rgb_array'))
                else: 
                    self.D['episode_sequence'].append(obs)
                self.D['action_sequence'].append(a)
                self.D['reward_sequence'].append(rew)
                self.D['done_sequence'].append(done)
                self.D['function_sequence'].append(self.model.forward(torch.FloatTensor(obs)))

                # TODO: Sample random minibatch of transitions from self.D and beyond
                # I think the minibatch is simply considers a transition of two consecutive timesteps.
                # It makes sense this way because it says from index j (considering it is a random number between 0 and self.timestep_size-1), 
                # to j+1 which is the next timestep.
                # This way the problem of dimensions at first iteration does not exist because we already have two frames! :D

                # TODO: Calculate y_j, which should be pretty straightforward by the formula given.

                # TODO: Perform gradient descend which I dunno how to do.
                # Help :D


                pass

        # TODO: Save the model. Not only the weights,
        # unless you remeber the configuration, 
        # because of the dynamic creation of the model

        self.env.close()


    def select_action(self, curr_timestep, policy='egreedy', epsilon=None, temp=None):
        # TODO: FIX! THIS IS JUST THE OLD IMPLEMENTATION. MUST TAKE VALUE FROM MODEL!
        # Instead of self.Q_sa we need to look at the model prediction in D['function_sequence']
        # 
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # Randomly generate a value between [0,1] with a uniform distribution
            p = np.random.uniform(0,1,1)[0]
            if p < epsilon:
                # Select random action
                a = np.random.randint(0,self.env.action_space.n)
            else:
                # Select most probable action
                a = round(self.D['function_sequence'][curr_timestep])  
                
        elif policy == 'softmax':
            # TODO: FIX
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # we use the provided softmax function in Helper.py
            probs = softmax(self.D['function_sequence'][curr_timestep], temp)
            a = np.random.choice(range(0, self.env.action_space.n),p=probs)
        return a




    