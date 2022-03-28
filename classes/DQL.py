import random
import numpy as np
import torch
from copy import deepcopy
from collections import deque
from PIL import Image
from Utilities import *
from classes.Model import *


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
            gamma = 1,
            policy = "egreedy",
            epsilon = None,
            temp = None,
            k = None,
            beta = None,
            eta = None,
            model = None,
            target_model = False,
            tm_wait = 10,
            input_is_img = False,
            custom_reward = False,
            env = None,
            render = False,
            device = None
        ):
        self.env = env
        if self.env is None:
            exit("Please select an environment")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Computing on {self.device} device")
        self.batch_size = batch_size
        self.use_rb = use_rb
        self.rb_size = rb_size
        self.n_episodes = n_episodes
        self.n_timesteps = n_timesteps
        self.gamma = gamma
        self.policy = policy
        # tensor of total count for every possible action
        self.actions_count = torch.tensor([1]*self.env.action_space.n, dtype=torch.float32, device=self.device)
        # tensor of total reward for every possible action
        self.actions_reward = torch.tensor([0]*self.env.action_space.n, dtype=torch.float32, device=self.device)
        self.epsilon = epsilon
        self.temp = temp
        self.k = k
        self.beta = beta
        self.eta = eta
        # used to track how many times a hashed state s is reaced (novelty based approach)
        self.state_hash = {}
        self.simhash_mat = None
        self.ICM = None
        self.model = model.to(self.device)
        # create an identical separated model updated as self.model each episode
        if target_model:
            self.target_model = deepcopy(self.model).to(self.device)
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
        self.custom_reward = custom_reward
        self.input_is_img = input_is_img
        self.render = render

    def __call__(self):
        # create replay buffer
        self.rb = deque([], maxlen=self.rb_size)

        # iterate over episodes
        self.training_started = False
        self.ts_tot = 0
        for ep in range(self.n_episodes):
            self.episode(ep)

        # TODO: Save the model. Not only the weights,
        # unless you remeber the configuration, 
        # because of the dynamic creation of the model

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_frame(self, frame, x_pos):
        # convert RGB frame to 32bit [0,1] greyscale
        frame = np.array(Image.fromarray(frame, "RGB").convert("L"), dtype=np.float32) / 255.0
        # cut top and bottom parts of the image, keeping 200 pixels in H
        # and cut width centering in the cart position, keeping 200 pixels in W
        pix_per_unit = int(300 / 2.4)
        pix_from_cent = int(x_pos*pix_per_unit)
        return frame[-250:-50, 300+pix_from_cent-100:300+pix_from_cent+100]

    def collect_frame(self, last_s):
        # collect RGB frame and preprocess it
        # s1_x is the x coord of the cart in the frame
        s1 = self.env.render(mode='rgb_array')
        s1_x = self.env.state[0]
        s1 = self.preprocess_frame(s1, s1_x)
        # stack the new and last frames in an array along depth and reshape to CxHxW
        if last_s is None:
            last_s = s1
        s = np.dstack((s1, last_s)).transpose((2, 0, 1))
        return s

    def self_sup_learn(self, mode=0):
        """ Three modalities:
            0 -> pretraining + finetuning
            1 -> pretraining
            2 -> finetuning
        """
        if mode == 2:
            print("Loading pretrained weights from self-supervised learning")
            # self.model.load_state_dict(torch.load("./ssl_pretrained/weights"))
            state_dict = torch.load("./ssl_pretrained/weights.pt")
            for name, param in state_dict.items():
                if "output_head" in name:
                    continue
                self.model.load_state_dict({name: param}, strict=False)

        else:  # pretraining
            print("Self-supervised pretraining started")
            # we use binary cross entropy as the loss function
            loss = torch.nn.BCELoss()
            # start self-supervised training
            self.model.train()
            training_steps = 0
            while True:
                done = False
                self.env.reset()
                target = self.collect_frame(None)
                while not done:
                    # collect frames and pass them through the autoencoder
                    target = self.collect_frame(target[0])
                    target = torch.tensor(target, device=self.device)
                    decoded_targ = self.model.forward_ssl(target.unsqueeze(0)).squeeze(0)
                    # compute loss and execute a training step
                    curr_loss = loss(decoded_targ, target)
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
                    optimizer.zero_grad()
                    curr_loss.backward()
                    self.optimizer.step()
                    # do a random action
                    _, _, done, _ = self.env.step(np.random.randint(0, 1))
                    training_steps += 1
                print("Loss:", curr_loss.cpu().detach().numpy())
                if training_steps >= 2500:
                    break

            # save evaluation samples
            with torch.no_grad():
                self.model.eval()
                out = self.model.forward_ssl(target.unsqueeze(0)).squeeze(0)
            Image.fromarray((np.array(target[0].cpu()*255)).astype(np.uint8)).save("og1.png")
            Image.fromarray((np.array(out[0].cpu().detach()*255)).astype(np.uint8)).save("dec1.png")

            # save model
            torch.save(self.model.state_dict(), "./ssl_pretrained/weights.pt")

            # freeze decoder weights
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def novelty_exploration(self, s):
        state_hash_key = str(list(np.sign(self.simhash_mat.dot(s))))
        if state_hash_key in self.state_hash.keys():
            self.state_hash[state_hash_key] += 1
        else:
            self.state_hash[state_hash_key] = 1
        return self.beta / np.sqrt(self.state_hash[state_hash_key])

    def curiosity_exploration(self, s, a, s_next):
        # train inverse model and freeze the forward model
        self.ICM.train()
        for param in self.ICM.feature_output.parameters():
            param.requires_grad = False
        for param in self.ICM.encoder.parameters():
            param.requires_grad = True
        for param in self.ICM.action_output.parameters():
            param.requires_grad = True
        # obtain the features fi(s) and predicted a
        s = torch.tensor(s, device=self.device).unsqueeze(0)
        feature_s, feature_s_next, a_pred = self.ICM.forward_inverse(s, s_next) 
        loss1 = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
        loss1(a_pred, a).backward()
        self.optimizer.step()
        # train the forward model and freeze inverse model
        for param in self.ICM.feature_output.parameters():
            param.requires_grad = True
        for param in self.ICM.encoder.parameters():
            param.requires_grad = False
        for param in self.ICM.action_output.parameters():
            param.requires_grad = False
        # getting the predicted s_next from the concatenation between the features fi(s) and a
        feature_s_forward = self.ICM.forward_feature(feature_s, a)
        loss2 = self.ICM.loss_feature(feature_s_forward, feature_s_next)
        self.optimizer.zero_grad()
        loss2.backward()
        self.optimizer.step()

        # return the intrinsic reward
        return self.eta * loss2.item() 

    def training_step(self):
        # draw batch of experiences from replay buffer
        sampled_exp = random.sample(self.rb, k=self.batch_size)
        s_exp, a_exp, r_exp, s_next_exp, done_exp = zip(*sampled_exp)
        s_exp = torch.stack(s_exp)
        s_next_exp = torch.stack(s_next_exp)
        a_exp = torch.stack(a_exp)
        r_exp = torch.stack(r_exp)
        done_exp = torch.stack(done_exp)
        # compute q values for current and next states using dnn
        self.model.train()
        q_exp = self.model.forward(s_exp).gather(1, a_exp.view(-1, 1)).view(-1)
        with torch.no_grad():
            self.target_model.eval()
            q_exp_target = self.target_model.forward(s_next_exp).detach().max(1)[0]
        # compute mean loss of the batch
        loss = self.loss(q_exp, r_exp + self.gamma*q_exp_target*~done_exp)
        # compute gradient of loss
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients in (-1, 1)
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # update weigths
        self.optimizer.step()
        
        return loss.cpu().detach().numpy()

    def episode(self, ep):
        if self.use_rb and ep == 0:
            print("Filling replay buffer before training...")
        # initialize starting state
        s = self.env.reset()
        if self.input_is_img:
            s = self.collect_frame(None)

        # TODO change from using self.policy values to other variable
        if self.policy == 'novelty-based' and self.simhash_mat == None:
            dim = list(s.shape)
            dim.insert(0, self.k)
            self.simhash_mat = np.random.normal(0, 1, tuple(dim))
        elif self.policy == 'curiosity-based' and self.ICM == None:
            self.ICM = ICM(s.shape[0], self.env.action_space.n)

        # iterate over timesteps
        loss_ep = 0
        ts_ep = 0
        r_ep = 0
        done = False
        while not done:
            if self.target_model != self.model and self.ts_tot % self.tm_wait:
                # update target model weigths as current self.model weights
                self.update_target()
            self.ts_tot, ts_ep = self.ts_tot + 1, ts_ep + 1
            # Select action using the behaviour policy
            a = self.select_action(s)
            if self.render:
                self.env.render()
            # Execute action a in emulator and observe reward r and next state s_next
            # The basic reward is always 1 (even if done is True)
            s_next, r, done, _ = self.env.step(a.item())
            if self.input_is_img:
                s_next = self.collect_frame(s[0])
            # TODO change from using self.policy values to other variable
            if self.policy == 'novelty-based':
                r += self.novelty_exploration(s)
            elif self.policy == 'curiosity-based':
                r += self.curiosity_exploration(s, a, s_next)
            # apply custom reward strategy
            if self.custom_reward:
                if done: r -= 1
                if ts_ep < 15: pass
                elif ts_ep < 50: r += 1
                elif ts_ep < 100: r += 2
                else:
                    r += ts_ep/100 + 2
            r_ep += r
            self.actions_reward[a] += r  # for ucb action selection
            # add experience to replay buffer (as torch tensors)
            self.rb.append((torch.tensor(s, device=self.device), a,
                torch.tensor(r, device=self.device), 
                torch.tensor(s_next, device=self.device),
                torch.tensor(done, device=self.device)))
            # set next state as the new current state
            s = s_next
            # to fill the replay buffer before starting training
            if self.use_rb and len(self.rb) < self.batch_size:
                continue
            elif not self.training_started:
                self.training_started = True
                print("Training started")
            # execute a training step on the DQN
            loss_ep += self.training_step()
            # if a maximum number of timesteps is set, check it
            if self.n_timesteps is not None and ts_ep == self.n_timesteps:
                break
        
        if not ((ep+1) % (1 if self.input_is_img else 50)):
            print(f"[{ep+1}] Episode mean loss: {round(loss_ep/ts_ep, 4)} | Episode reward: {r_ep} | Timesteps: {ts_ep}")

    def select_action(self, s):
        """ Select a behaviour policy between epsilon-greedy, softmax (boltzmann) and upper confidence bound
        """
        with torch.no_grad():
            self.model.eval()
            q_values = self.model.forward(torch.tensor(s, device=self.device).unsqueeze(0))

        if self.policy == "egreedy":
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            # annealing of epsilon
            if self.epsilon.__class__.__name__ == "tuple":  # exponential annealing
                epsilon = self.epsilon[0] + (self.epsilon[1] - self.epsilon[0]) * np.exp(-1. * self.ts_tot / self.epsilon[2])
            else:  # no annealing
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
            Ut = 3 * torch.sqrt(torch.log(torch.tensor(self.ts_tot)) / self.actions_count)
            a = argmax(Qt + Ut)
            self.actions_count[a] += 1

        else:
            exit("Please select an existent behaviour policy")
        
        return a.to(self.device)