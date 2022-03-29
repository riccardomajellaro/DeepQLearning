#!/bin/bash
# Script for simple mlp network, without images
python experiment.py \
-net 'mlp' \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 10000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 50 \
-custom_reward \
-policy egreedy \
-epsilon 0.02 0.99 100. \
-run_name basic_mlp

python experiment.py \
-net 'mlp' \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 10000 \
-batch_size 256 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 20 \
-custom_reward \
-policy egreedy \
-epsilon 0.02 0.99 1000.
-run_name basic_mlp_2