#!/bin/bash
# Script for simple mlp network, without images
python experiment.py \
-net 'mlp' \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 10000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 5 \
-custom_reward \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name basic_mlp

python experiment.py \
-net 'mlp' \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 256 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-custom_reward \
-policy egreedy \
-epsilon 0.02 0.99 400.
-run_name basic_mlp_2