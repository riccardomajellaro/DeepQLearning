import torch
import gym
# from classes.DQL_old import DQL
from classes.DQL import DQL
from classes.Model import *
from Utilities import argmax

use_img = True
evaluate = True
device = None

env = gym.make('CartPole-v1')

# net = NN(4, 2, 1, 8)
# net = MLP(4, 2)
net = ConvNet(200, 200, 2, 2)

loss = torch.nn.SmoothL1Loss()
# loss = torch.nn.L1Loss()
# loss = torch.nn.MSELoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-2)
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-2)

dql = DQL(
    rb_size=10000, batch_size=128, n_episodes=50, device=device,
    loss=loss, optimizer=optimizer, gamma=0.9,
    policy="egreedy", epsilon=(0.05, 0.9, 200), temp=0.1,
    model=net, target_model=True, tm_wait=10,
    custom_reward=True, env=env, render=False, input_is_img=use_img
)
dql()

if evaluate:
    from time import sleep
    done = False
    env.reset()
    env.render()
    while not done:
        if use_img:
            s = dql.collect_frames()
        else:
            s = env.state
        with torch.no_grad():
            s, _, done, _ = env.step(int(argmax(net.forward(torch.tensor(s, dtype=torch.float32, device=dql.device).unsqueeze(0)))))
        env.render()
        sleep(0.2)

env.close()