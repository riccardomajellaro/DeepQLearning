# from classes.DQL_old import DQL
from classes.DQL import DQL
from classes.Model import *
import gym
import torch
from Utilities import argmax

evaluate = False
device = None

env = gym.make('CartPole-v1')
# net = NN(4, 2, 1, 8)
net = MLP(4, 2)
loss = torch.nn.SmoothL1Loss()
# loss = torch.nn.L1Loss()
# loss = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-2)
# optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3)
dql = DQL(
    rb_size=10000, batch_size=128, n_episodes=3000, device=device,
    loss=loss, optimizer=optimizer, gamma=0.99,
    policy="egreedy", epsilon=(0.05, 0.9, 600), temp=0.1,
    model=net, target_model=True, tm_wait=10,
    custom_reward=True, env=env, render=False
)
dql()

if evaluate:
    done = False
    s = env.reset()
    while not done:
        env.render()
        with torch.no_grad():
            s, _, _, done = env.step(int(argmax(net.forward(torch.tensor(s, device=device)))))
    env.reset()