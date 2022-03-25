from classes.DQL import DQL
from classes.Model import *
import gym
import torch
from Utilities import argmax

evaluate = False

env = gym.make('CartPole-v1')
# net = NN(4, 2, 1, 8)
net = MLP(4, 2)
# loss = torch.nn.SmoothL1Loss()
# loss = torch.nn.L1Loss()
loss = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5)
dql = DQL(
    rb_size=10000, batch_size=256, n_episodes=3000,
    loss=loss ,optimizer=optimizer, gamma=0.99,
    policy="egreedy", epsilon=(0.05, 0.9, 400), temp=0.1,
    model=net, target_model=True, env=env, render=False
)
dql()

if evaluate:
    done = False
    s = env.reset()
    while not done:
        env.render()
        s, _, _, done = env.step(argmax(net.forward(torch.FloatTensor(s))))
    env.reset()