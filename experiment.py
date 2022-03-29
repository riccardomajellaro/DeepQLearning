import torch
import gym
from classes.DQL import DQL
from classes.Model import *
from Utilities import argmax

use_img = True
if not use_img: ssl_mode = None
else: ssl_mode = None  # None: no ssl, 0: pretrain+finetune, 1: pretrain, 2: finetune
evaluate = True

env = gym.make('CartPole-v1')

# net = NN(4, 2, 1, 8)
# net = MLP(4, 2)
net = ConvNet(2, 2, dueling=True)
# net = SSLConvNet(2, 2, dueling=True)

loss = torch.nn.SmoothL1Loss()
# loss = torch.nn.L1Loss()
# loss = torch.nn.MSELoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  #, weight_decay=1e-2)
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-2)

# TODO AGGIUNGI MODALITA' NO REPLAY BUFFER DIOCANE
# TODO add assert statements for not allowing some configs together
dql = DQL(
    rb_size=10000, batch_size=128, n_episodes=3000, device="cuda",
    loss=loss, optimizer=optimizer, gamma=0.99, intr_rew="curiosity-based",
    policy="egreedy", epsilon=(0.05, 0.9, 200), temp=0.1, k=16, beta=0.2,
    eta=0.6, model=net, target_model=True, tm_wait=10, double_dql=True,
    custom_reward=True, env=env, render=False, input_is_img=use_img
)

if ssl_mode is not None:
    dql.self_sup_learn(ssl_mode)
if ssl_mode is None or ssl_mode in [0, 2]:
    dql()

if evaluate:
    from time import sleep
    done = False
    s = env.reset()
    env.render()
    if use_img:
        s = dql.collect_frame(None)
    input("Press enter to start the evaluation...")
    while not done:
        with torch.no_grad():
            net.eval()
            s_next, _, done, _ = env.step(int(argmax(net.forward(torch.tensor(s, dtype=torch.float32, device=dql.device).unsqueeze(0)))))
            if use_img:
                s_next = dql.collect_frame(s[0])
            s = s_next
        env.render()
        sleep(0.1)

env.close()