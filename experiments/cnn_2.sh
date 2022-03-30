#!/bin/bash
# Script for simple mlp network, without images
python experiment.py \
-use_img \
-net cnn \
-dueling \
-loss smooth_l1 \
-optimizer rms \
-optim_lr 1e-4 \
-rb_size 2500 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.95 \
-target_model \
-tm_wait 5 \
-custom_reward \
-intr_rew curiosity-based \
-policy egreedy \
-epsilon 0.025 0.99 200. \
-run_name cnn_2 ^