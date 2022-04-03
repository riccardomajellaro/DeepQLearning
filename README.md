This repository contains the solution to the second assignment of the course Reinforcement Learning.
# Requirements
 
 `pip install -r requirements.txt`

# How to train all the configurations

All the experiments presented in the report are fully repruducible by running the command
`./experiments/basic_exps.sh` from the main folder of the repository. It is important to run the script out of the directory `experiments` using the command above to avoid errors with absoulute and relative paths. 
Furthermore, it is important to change the script permissions in order to make it executable as a program.

# How to train a configurations
`python experiment.py`
along with the arguments that are already used in `basic_exps.sh`
# How to evaluate a configuration
Run the command below from the main directory
`python evaluate.py`
along with the following arguments:
<ul>
<li>`-net MODEL`, where MODEL can be "mlp", "cnn", "ssl_cnn", "tl_cnn" </li>
<li>`-use_img`, in case of Visual CartPole</li>
<li>`-run_name`, name of the config to run </li>
<li>`-render`, to visualize the environment</li>
<li>`-device`, to indicate where to execute the computations (e.g. "cpu" or "cuda") </li>
<li>`-virtual_display`, to execute for example on a headless server</li>
</ul>
