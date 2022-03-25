import numpy as np
import torch
from copy import deepcopy
from collections import deque
from Utilities import *


class DQL:
    """ Parameters:
            - use_rb : enables DQN with experience replay buffer
            - rb_size : experience replay buffer size
            - n_episodes : size of the episode to consider
            - n_timesteps : numer of timesteps per episode to consider
            - minibatch_size : size of minibatch to consider
            - epsilon : probability to choose a random action
            - temp : temperature parameter for softmax selection
            - model : Deep Learning agent (model in pytorch)
            - input_is_img : defines if the input is an image
            - env : Environment to train our model 

        About the model:
            The model should output two parameters which indicates the probability
            for each action. The cart problem can be solved with the sigmoid
            and only one output on the model however, we will do it this way 
            in order to handle different environments with more actions later on.
    """
    def __init__(
            self, use_rb=True,
            batch_size = 5,
            rb_size = 20,
            n_episodes = 100,
            n_timesteps = None,
            optimizer=None,
            policy = "egreedy",
            epsilon = None,
            temp = None,
            gamma = 1,
            model = None,
            target_model = False,
            input_is_img = False,
            env = None,
            render = False
        ):
        self.env = env
        if self.env is None:
            exit("Please select an environment")
        self.batch_size = batch_size
        self.use_rb = use_rb
        self.rb_size = rb_size
        self.n_episodes = n_episodes
        self.n_timesteps = n_timesteps
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        self.gamma = gamma
        self.model = model
        # create an identical separated model updated as self.model each episode
        if target_model:
            self.target_model = deepcopy(self.model)
        # target_model is exactly self.model
        else:
            self.target_model = self.model
        self.optimizer = optimizer
        if self.optimizer is None:
            exit("Please select an optimization algorithm")
        self.input_is_img = input_is_img
        self.render = render

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def __call__(self):
        # Create replay buffer
        self.rb = deque(maxlen=self.rb_size)

        # Iterate over episodes
        training_started = False
        for ep in range(self.n_episodes):
            if self.use_rb and ep == 0:
                print("Filling replay buffer before training...")
            # Initialize sequence s1 = {x1} and preprocess f1 = f(s1)
            s = self.env.reset()
            # Control if we are using images as input of the model instead of observations
            if self.input_is_img:
                # TODO: add normalization. 
                # (also flattening is needed for DNN)
                # but it can be done in the model when creating it which is better
                s = self.env.render(mode='rgb_array')

            # update target model weigths as to current self.model weights
            if self.target_model:
                self.update_target()
            
            # Iterate over timesteps
            loss_tot = 0
            ts_tot = 0
            r_tot = 0
            done = False
            while not done:
                ts_tot += 1
                # Select random action with probability epsilon or follow egreedy policy
                a = self.select_action(s, self.model.forward(torch.FloatTensor(s)))
                if self.render:
                    self.env.render()
                # Execute action a_t in emulator and observe reward rt and image x_t+1
                s_next, r, done, _ = self.env.step(a)
                r_tot += r
                # Save step in replay buffer
                if self.input_is_img:
                    s_next = self.env.render(mode='rgb_array')
                # add to replay buffer
                self.rb.append((s, a, r, s_next, done))
                # draw from replay buffer
                if self.use_rb and len(self.rb) < self.rb_size:  # to fill the replay buffer before starting training
                    continue
                elif not training_started:
                    training_started = True
                    print("Training started")
                sampled_exp = np.array(self.rb, dtype=object)[np.random.choice(len(self.rb), size=self.batch_size)]
                s_exp = torch.FloatTensor(np.array([sample[0] for sample in sampled_exp]))
                a_exp = [sample[1] for sample in sampled_exp]
                done_exp = [sample[4] for sample in sampled_exp]
                r_exp = torch.FloatTensor([sample[2] if not done_exp[i] else 0 for i, sample in enumerate(sampled_exp)])
                s_next_exp = torch.FloatTensor(np.array([sample[3] for sample in sampled_exp]))
                # compute q values for target and current using dnn
                q_exp = self.model.forward(s_exp)[np.arange(len(a_exp)), a_exp]
                q_exp_target = torch.max(self.target_model.forward(s_next_exp), axis=1)[0]
                # compute loss
                loss = torch.mean((r_exp + self.gamma*q_exp_target - q_exp)**2)
                loss_tot += loss.detach().numpy()
                # compute gradient of loss
                self.model.train()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.n_timesteps is not None and ts_tot == self.n_timesteps:
                    break
            
            if not ((ep+1) % 50):
                print(f"[{ep+1}] Episode mean loss: {loss_tot/ts_tot} | Episode reward: {r_tot}")

        # TODO: Save the model. Not only the weights,
        # unless you remeber the configuration, 
        # because of the dynamic creation of the model

        self.env.close()

    def select_action(self, s, q_values):
        """ Select a behaviour policy between epsilon-greedy and softmax (boltzmann)
        """
        if self.policy == "egreedy":
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # Randomly generate a value between [0,1] with a uniform distribution
            if np.random.uniform(0, 1) < self.epsilon:
                # Select random action
                a = np.random.randint(0, self.env.action_space.n)
            else:
                # Select most probable action
                a = argmax(q_values)
                
        elif self.policy == "softmax":
            if self.temp is None:
                raise KeyError("Provide a temperature")

            # we use the provided softmax function in Helper.py
            probs = softmax(q_values, self.temp)
            a = np.random.choice(range(0, self.env.action_space.n), p=probs)
        
        return a