from gc import collect
import random
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
            loss=None,
            optimizer=None,
            policy = "egreedy",
            epsilon = None,
            temp = None,
            gamma = 1,
            model = None,
            target_model = False,
            tm_wait = 10,
            input_is_img = False,
            env = None,
            render = False,
            device = None
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
        # tensor of total count for every possible action
        self.actions_count = torch.tensor([1]*self.env.action_space.n, dtype=torch.float64)
        # tensor of total reward for every possible action
        self.actions_reward = torch.tensor([0]*self.env.action_space.n, dtype=torch.float64)
        self.epsilon = epsilon
        self.temp = temp
        self.gamma = gamma
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Computing on {self.device} device")
        self.model = model.to(self.device)
        # create an identical separated model updated as self.model each episode
        if target_model:
            self.target_model = deepcopy(self.model).to(self.device)
            self.target_model.eval()
        # target_model is exactly self.model
        else:
            self.target_model = self.model
        self.tm_wait = tm_wait
        self.loss = loss
        if self.loss is None:
            exit("Please select a loss")
        self.optimizer = optimizer
        if self.optimizer is None:
            exit("Please select an optimization algorithm")
        self.input_is_img = input_is_img
        self.render = render

    def __call__(self):
        # Create replay buffer
        self.rb = deque([], maxlen=self.rb_size)

        # Iterate over episodes
        training_started = False
        ts_tot = 0
        for ep in range(self.n_episodes):
            if self.use_rb and ep == 0:
                print("Filling replay buffer before training...")
            # Initialize starting state
            s = self.env.reset()
            # Control if we are using images as input of the model instead of observations
            if self.input_is_img:
                # TODO: add normalization. 
                # (also flattening is needed for DNN)
                # but it can be done in the model when creating it which is better
                s = self.env.render(mode='rgb_array')
            
            # Iterate over timesteps
            loss_ep = 0
            ts_ep = 0
            r_ep = 0
            done = False
            while not done:
                if self.target_model != self.model and ts_tot % self.tm_wait:
                    # update target model weigths as current self.model weights
                    self.update_target()
                ts_tot, ts_ep = ts_tot + 1, ts_ep + 1
                # Select random action with probability epsilon or follow egreedy policy
                self.model.eval()
                a = self.select_action(self.model.forward(torch.tensor(s).unsqueeze(0), self.device), ts_tot)
                if self.render:
                    self.env.render()
                # Execute action a in emulator and observe reward r and next state s_next
                s_next, r, done, _ = self.env.step(a.item())
                r_ep += r
                self.actions_reward[a] += r
                # Save step in replay buffer
                if self.input_is_img:
                    s_next = self.env.render(mode='rgb_array')
                # add to replay buffer
                self.rb.append((s, a, r, s_next, done))
                # set current state as next_state
                s = s_next
                # to fill the replay buffer before starting training
                if self.use_rb and len(self.rb) < self.batch_size:
                    continue
                elif not training_started:
                    training_started = True
                    print("Training started")
                # draw batch from replay buffer
                # samples_indexes = np.random.choice(len(self.rb), size=self.batch_size)
                # print(self.rb)
                # sampled_exp = torch.tensor(self.rb)[samples_indexes]
                sampled_exp = random.sample(self.rb, k=self.batch_size)
                s_exp = torch.tensor(np.array([sample[0] for sample in sampled_exp]))
                a_exp = [sample[1] for sample in sampled_exp]
                done_exp = torch.tensor([sample[4] for sample in sampled_exp])
                r_exp = torch.tensor([sample[2] if not done_exp[i] else 0 for i, sample in enumerate(sampled_exp)])
                s_next_exp = torch.tensor(np.array([sample[3] for sample in sampled_exp]))
                # compute q values for target and current using dnn
                self.model.train()
                q_exp = self.model.forward(s_exp, self.device)[np.arange(len(a_exp)), a_exp]
                q_exp_target = torch.max(self.target_model.forward(s_next_exp, self.device), axis=1)[0].detach()
                # compute mean loss of the batch
                loss = self.loss(q_exp, r_exp.to(self.device) + self.gamma*q_exp_target*~done_exp.to(self.device))
                loss_ep += loss.cpu().detach().numpy()
                # compute gradient of loss
                self.optimizer.zero_grad()
                loss.backward()
                # clip gradients in (-1, 1)
                # for param in self.model.parameters():
                #     param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                # if a maximum number of timesteps is set, check it
                if self.n_timesteps is not None and ts_ep == self.n_timesteps:
                    break
            
            if not ((ep+1) % 100):
                print(f"[{ep+1}] Episode mean loss: {round(loss_ep/ts_ep, 4)} | Episode reward: {r_ep} | Timesteps: {ts_ep}")

        # TODO: Save the model. Not only the weights,
        # unless you remeber the configuration, 
        # because of the dynamic creation of the model

        self.env.close()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, q_values, t):
        """ Select a behaviour policy between epsilon-greedy, softmax (boltzmann) and upper confidence bound
        """
        if self.policy == "egreedy":
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            # annealing of epsilon
            if self.epsilon.__class__.__name__ == "tuple":
                epsilon = self.epsilon[1] + (self.epsilon[0] - self.epsilon[1]) * np.exp(-1. * t / self.epsilon[2])
            else:
                epsilon = self.epsilon
            # Randomly generate a value between [0,1] with a uniform distribution
            if np.random.uniform(0, 1) < epsilon:
                # Select random action
                a = torch.tensor(np.random.randint(0, self.env.action_space.n), dtype=torch.int64)
            else:
                # Select most probable action
                a = argmax(q_values)
                
        elif self.policy == "softmax":
            if self.temp is None:
                raise KeyError("Provide a temperature")

            # we use the provided softmax function in Helper.py
            probs = softmax(q_values, self.temp).detach().numpy()
            a = np.random.choice(range(0, self.env.action_space.n), p=probs)

        elif self.policy == "ucb":
            Qt = self.actions_reward / self.actions_count
            Ut = 3 * torch.sqrt(torch.log(torch.tensor(t)) / self.actions_count)
            a = argmax(Qt + Ut)
            self.actions_count[a] += 1

        else:
            exit("Please select an existent behaviour policy")
        
        return a