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
        # tensor of total count for every possible action
        self.actions_count = torch.tensor([1]*self.env.action_space.n, dtype=torch.float64)
        # tensor of total reward for every possible action
        self.actions_reward = torch.tensor([0]*self.env.action_space.n, dtype=torch.float64)
        self.epsilon = epsilon
        self.temp = temp
        self.gamma = gamma
        self.model = model
        # create an identical separated model updated as self.model each episode
        if target_model:
            self.target_model = deepcopy(self.model)
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

    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def __call__(self):
        # Create replay buffer
        self.rb = deque([], maxlen=self.rb_size)

        # Iterate over episodes
        training_started = False
        ts_tot = 0
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
                a = self.select_action(self.model.forward(torch.tensor(s)), ts_tot)
                if self.render:
                    self.env.render()
                # Execute action a_t in emulator and observe reward rt and image x_t+1
                s_next, r, done, _ = self.env.step(a)
                r_ep += r
                self.actions_reward[a] += r
                # Save step in replay buffer
                if self.input_is_img:
                    s_next = self.env.render(mode='rgb_array')
                # add to replay buffer
                self.rb.append((s, a, r, s_next, done))
                # to fill the replay buffer before starting training
                if self.use_rb and len(self.rb) < self.batch_size:
                    continue
                elif not training_started:
                    training_started = True
                    print("Training started")
                # draw batch from replay buffer
                sampled_exp = np.array(self.rb, dtype=object)[np.random.choice(len(self.rb), size=self.batch_size)]
                s_exp = torch.tensor(np.array([sample[0] for sample in sampled_exp]))
                a_exp = [sample[1] for sample in sampled_exp]
                done_exp = torch.tensor([sample[4] for sample in sampled_exp])
                r_exp = torch.tensor([sample[2] if not done_exp[i] else 0 for i, sample in enumerate(sampled_exp)])
                s_next_exp = torch.tensor(np.array([sample[3] for sample in sampled_exp]))
                # compute q values for target and current using dnn
                q_exp = self.model.forward(s_exp)[np.arange(len(a_exp)), a_exp]
                q_exp_target = torch.max(self.target_model.forward(s_next_exp), axis=1)[0].detach()
                # compute loss
                loss = self.loss(q_exp, r_exp + self.gamma*q_exp_target*done_exp)
                loss_ep += loss.detach().numpy()
                # compute gradient of loss
                self.model.train()
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                if self.n_timesteps is not None and ts_ep == self.n_timesteps:
                    break
            
            if not ((ep+1) % 100):
                print(f"[{ep+1}] Episode mean loss: {loss_ep/ts_ep} | Episode reward: {r_ep}")

        # TODO: Save the model. Not only the weights,
        # unless you remeber the configuration, 
        # because of the dynamic creation of the model

        self.env.close()

    def select_action(self, q_values, t):
        """ Select a behaviour policy between epsilon-greedy and softmax (boltzmann)
        """
        if self.policy == "egreedy":
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            # annealing of epsilon
            if self.epsilon.__class__.__name__ == "tuple":
                epsilon = self.epsilon[1] + (self.epsilon[0] - self.epsilon[1]) * np.exp(-1. * t / self.epsilon[2])
            # Randomly generate a value between [0,1] with a uniform distribution
            if np.random.uniform(0, 1) < epsilon:
                # Select random action
                a = np.random.randint(0, self.env.action_space.n)
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
            Ut = 2 * torch.sqrt(torch.log(torch.tensor(t)) / self.actions_count)
            a = argmax(Qt + Ut)
            self.actions_count[a] += 1

        else:
            exit("Please select an existent behaviour policy")
        
        return a