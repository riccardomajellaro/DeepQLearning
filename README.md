# Solving (Vision) CartPole V1 with Deep Q Learning
This repository contains the solution to the second assignment of the course Reinforcement Learning from Leiden University. The CartPole V1 challenge from OpenAI was solved by using Deep Q Learning, both in the standard scenario with a 4-dimensional state space and in the vision scenario with frames as states. For additional information regarding the problem and the methodologies involved in its resolution, please refer to the <a href=https://github.com/riccardomajellaro/DeepQLearning/blob/main/report_extract.pdf>extract from our report</a>.

## Authors
<a href="https://github.com/OhGreat">Dimitrios Ieronymakis</a>, <a href="https://github.com/JonathanCollu">Jonathan Collu</a> and <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>

## Requirements
 
```
 pip install -r requirements.txt
```
## Usage

### Reproduce our experiments

All the experiments presented in the report are fully reproducible by running the script `./experiments/run_all_exp.sh`. It is important to run the above script from the main directory, to avoid errors with absolute and relative paths.


### Training a new configuration

To train a configuration you can use the following command, along with the available arguments described below:

```
 python experiment.py
```


Model parameters:
- `-use_img` : use this flag to define using frames as states.
- `-ssl_mode` : defines the use of self-supervised learning. *0*: pretraining using SSL + fine-tuning using DQL; *1*: only pretraining using SSL; *2*: only fine-tuning using DQL. 
- `-tl_mode` : defines the use of transfer learning. Equal to the `-ssl_mode` above.
- `-evaluate` : set this flag to evaluate the model at the end of the training step.
- `-run_name` : name of the experiment used to save model weights and statistics. 
- `-net` : defines the actual model to use.
- `-loss` : defines the loss function to use while training the model.
- `-optimizer` : defines the optimizer to use for training the model.
- `-optim_lr` : defines the learning rate for the optimizer.
- `-device` : defines the device on which to run the algorithm. This parameter is relative to the physical devices available on your pytorch environment. 

DQL parameters:
- `-rb_size` : defines the replay buffer size.
- `-batch_size` : defines the batch size.
- `-n_episodes` : defines the number of episodes for which to train the model.
- `-gamma` : defines the discount factor of previous rewards.
- `-target_model` : defines the usage of a target model during training.
- `-tm_wait` : defines the number of timesteps to wait before updating the target model.
- `-double_dqn` : defines the usage of the double DQN strategy.
- `-dueling` :  defines the usage of the dueling strategy.
- `-custom_reward` : defines the usage of customly weighted rewards during training. 
- `-intr_rew` : defines the usage of intrinsec rewards strategy.
- `-policy` : defines the policy of choice for the model.
- `-epsilon` : defines the epsilon value for the egreedy policy. Can be a float or a collection of 3 floats that specify the exponential annealing parameters.
- `-temp` : defines the temperature parameter for the softmax policy.
- `-k` : defines the *k* parameter for the intrinsec rewards of the novelty based exploration.
- `-beta` : defines the *beta* parameter for the intrinsec rewards of the novelty based exploration.
- `-eta` : defines the *eta* parameter for the intrinsec rewards of the curiosity based exploration.
- `-render` : use this flag to visualize the agent during training.
- `-virtual_display` : use this flag when training the model on a headless server. Avoids errors of unavailable display for pygame.


### Evaluating a configuration
To evaluate a configuration run the command below from the main directory:
```
 python evaluate.py
```
along with the following provided arguments:
- `-net MODEL` : where MODEL can be "mlp", "cnn", "ssl_cnn", "tl_cnn"
- `-use_img` : in case of Vision CartPole
- `-run_name` : name of the config to run
- `-render` : to visualize the environment
- `-device` : to indicate where to execute the computations (e.g. "cpu" or "cuda")
- `-virtual_display` : to execute for example on a headless server


## Results
It is possible to train a perfect agent (500 steps per episode) for both standard and vision CartPole. The complete results contained in our full report are omitted in this repository. The following GIF serves as a simple demonstration from one of our experiments (Vision CartPole, CNN + Curiosity configuration):

 ![](https://github.com/riccardomajellaro/DeepQLearning/blob/main/readme_files/cartpole_solved_expampl.gif)
