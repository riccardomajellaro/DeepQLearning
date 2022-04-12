# Solving (Vision) CartPole V1 with Deep Q Learning
This repository contains the solution to the second assignment of the course Reinforcement Learning from Leiden University. We solved the CartPole V1 challenge from OpenAI using Deep Q Learning, both in the standard scenario with a 4-dimensional state space and in the vision scenario with frames as states. For additional information regarding the problem and the methodologies involved in its resolution, please refer to the <a href=https://github.com/riccardomajellaro/DeepQLearning/blob/main/report_extract.pdf>extract from our report<a/>.

## Authors
<a href="https://github.com/OhGreat">Dimitrios Ierinomakys</a>, <a href="https://github.com/JonathanCollu">Jonathan Collu</a> and <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>

## Requirements
 
```
 pip install -r requirements.txt
```

## How to train all the configurations

All the experiments presented in the report are fully reproducible by running the command
`./experiments/run_all_exp.sh` from the main folder of the repository. It is important to run the script out of the directory `experiments` using the command above to avoid errors with absolute and relative paths.

## How to train a configuration
```
 python experiment.py
```
along with the arguments that are already used in `run_all_exp.sh`

A list and description of all the arguments will be provided in the future.

## How to evaluate a configuration
Run the command below from the main directory
```
 python evaluate.py
```
along with the following arguments:
- `-net MODEL`, where MODEL can be "mlp", "cnn", "ssl_cnn", "tl_cnn"
- `-use_img`, in case of Vision CartPole
- `-run_name`, name of the config to run
- `-render`, to visualize the environment
- `-device`, to indicate where to execute the computations (e.g. "cpu" or "cuda")
- `-virtual_display`, to execute for example on a headless server

## Results
It is possible to train a perfect agent (500 steps per episode) for both standard and vision CartPole. The complete results contained in our full report are omitted in this repository. The following GIF serves as a simple demonstration from one of our experiments (Vision CartPole, CNN + Curiosity configuration):

 ![](https://github.com/riccardomajellaro/DeepQLearning/blob/main/readme_files/cartpole_solved_expampl.gif)
