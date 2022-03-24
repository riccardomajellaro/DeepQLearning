import numpy as np
import gym

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
                        model = None,
                        env = None,
                ):
        self.experiece_replay = replay_buffer
        self.er_size = er_size
        self.episode_size = episode_size
        self.timestep_size = timesteps_size
        self.model = model
        self.env = env
        

    def __call__(self):

        env = gym.make(self.env)
        env.reset()
        episode_sequence = []



        # Iterate over episodes
        for i in range(self.episode_size):
            # Initialize sequence s1 = {x1} and preprocess f1 = f(s1)
            s1 = env.render(mode='rgb_array')
            episode_sequence.append(s1)



    