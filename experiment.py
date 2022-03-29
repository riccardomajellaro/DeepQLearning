from asyncio import run
from xmlrpc.client import Boolean
import torch
import gym
from classes.DQL import DQL
from classes.Model import *
from Utilities import argmax
import argparse

def main():

    parser = argparse.ArgumentParser()
    # Parse model parameters
    parser.add_argument('-use_img', action='store_true') # True only when used (store_true)
    parser.add_argument('-ssl_mode', action='store', type=int, default=None)
    parser.add_argument('-evaluate', action='store_true')
    parser.add_argument('-run_name', action='store', type=str, default=None)
    parser.add_argument('-net', action='store', type=str, default=None)
    parser.add_argument('-loss', action='store', type=str, default='mse')
    parser.add_argument('-optimizer', action='store', type=str, default='adam')
    parser.add_argument('-optim_lr', action='store', type=float, default=1e-3)
    parser.add_argument('-device', action='store', type=str, default="cuda")

    # Parse DQN parameters
    parser.add_argument('-rb_size', action='store', type=int, default=10000)
    parser.add_argument('-batch_size', action='store', type=int, default=128)
    parser.add_argument('-n_episodes', action='store', type=int, default=10000)
    parser.add_argument('-gamma', action='store', type=float, default=0.99)
    parser.add_argument('-target_model', action='store_true')
    parser.add_argument('-tm_wait', action='store', type=int, default=10)
    parser.add_argument('-double_dql', action='store_true')
    parser.add_argument('-custom_reward', action='store_true')
    parser.add_argument('-intr_rew', action='store', type=str, default=None)
    parser.add_argument('-policy', action='store', type=str, default=None)
        # Remember to pass epsilon values as floats
    parser.add_argument('-epsilon', action='store', type=float, 
                        nargs="+", default=[0.02, 0.99, 200.])
    parser.add_argument('-temp', action='store', type=float, default=0.1)
    parser.add_argument('-k', action='store', type=int, default=16)
    parser.add_argument('-beta', action='store', type=float, default=0.02)
    parser.add_argument('-eta', action='store', type=float, default=0.6)
    parser.add_argument('-render', action='store_true')
    """
    double_dql bool
    custom_reward bool
    intr_rew string or none
    policy str
    epsilon numer or tuple (make list and convert)
    temp float
    k float
    beta float
    eta float
    render bool
    """
    args = parser.parse_args()
    print("arguments passed:",args)

    env = gym.make('CartPole-v1')

    losses = {  'mse' : torch.nn.MSELoss,
                'l1' : torch.nn.L1Loss,
                'smooth_l1' : torch.nn.SmoothL1Loss()
            }
    optimizers = {  'adam': torch.optim.Adam,
                    'sgd': torch.optim.SGD,
                    'rms': torch.optim.RMSprop
                }

    # Initialize model params, loss and optimizer
    use_img = args.use_img
    if not use_img: ssl_mode = None
    else: ssl_mode = args.ssl_mode  # None: no ssl, 0: pretrain+finetune, 1: pretrain, 2: finetune
    if args.net == 'mlp':
        net = MLP(4, 2)
    elif args.net == 'conv':
        net = ConvNet(2, 2, dueling=True)
    elif args.net == 'ssl_conv':
        net = SSLConvNet(2, 2, dueling=True)
    else:
        print('Select a correct network')
        exit()
    loss = losses[args.loss]()
    optimizer = optimizers[args.optimizer](net.parameters(), args.optim_lr)

    # Extra control which also sets batch size to 1 when
    # we don't want to use a buffer, we simply need to set rb_size = 1 .
    batch_size = min(args.rb_size, args.batch_size)

    # Fix epsilon as tuple
    if len(args.epsilon) > 1:
        epsilon = tuple(args.epsilon)
    else: epsilon = args.epsilon[0]
    print(f"epsilon: {epsilon}")

    # Run name + directory
    run_name = None if args.run_name == None else "array_results/"+run_name

    # TODO add assert statements for not allowing some configs together
    dql = DQL(
        rb_size=args.rb_size, batch_size=batch_size, n_episodes=args.n_episodes, 
        device=args.device, loss=loss, optimizer=optimizer, gamma=args.gamma, 
        policy=args.policy, epsilon=(0.05, 0.9, 200), temp=args.temp, k=args.k, beta=args.beta,
        eta=0.6, model=net, target_model=args.target_model, tm_wait=args.tm_wait, 
        double_dql=args.double_dql, intr_rew=args.intr_rew, custom_reward=args.custom_reward, 
        env=env, input_is_img=use_img, render=args.render, 
        run_name=run_name
    )

    if ssl_mode is not None:
        dql.self_sup_learn(ssl_mode)
    if ssl_mode is None or ssl_mode in [0, 2]:
        dql()

    # Test an evaluation run after the model is done training
    if args.evaluate:
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

if __name__ == "__main__":
    main()