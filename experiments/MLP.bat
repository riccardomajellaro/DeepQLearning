python experiment.py ^
-net mlp ^
-loss mse ^
-optimizer adam ^
-optim_lr 1e-3 ^
-rb_size 1 ^
-batch_size 1 ^
-n_episodes 1000 ^
-gamma 0.99 ^
-policy egreedy ^
-epsilon 0.02 0.99 200. ^
-run_name C:/Users/ricca/Desktop/exp_results/MLP_base

python experiment.py ^
-net mlp ^
-loss mse ^
-optimizer adam ^
-optim_lr 1e-3 ^
-rb_size 100000 ^
-batch_size 64 ^
-n_episodes 1000 ^
-gamma 0.99 ^
-policy egreedy ^
-epsilon 0.02 0.99 200. ^
-run_name C:/Users/ricca/Desktop/exp_results/MLP_buffer_64

python experiment.py ^
-net mlp ^
-loss mse ^
-optimizer adam ^
-optim_lr 1e-3 ^
-rb_size 10000 ^
-batch_size 128 ^
-n_episodes 1000 ^
-gamma 0.99 ^
-policy egreedy ^
-epsilon 0.02 0.99 200. ^
-run_name C:/Users/ricca/Desktop/exp_results/MLP_buffer_128_10000

python experiment.py ^
-net mlp ^
-loss mse ^
-optimizer adam ^
-optim_lr 1e-3 ^
-rb_size 1000 ^
-batch_size 128 ^
-n_episodes 1000 ^
-gamma 0.99 ^
-policy egreedy ^
-epsilon 0.02 0.99 200. ^
-run_name C:/Users/ricca/Desktop/exp_results/MLP_buffer_128_1000