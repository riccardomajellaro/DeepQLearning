#!/bin/bash

# MLP base 
echo "~~~MLP base~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 1 \
-batch_size 1 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_base ;

# MLP + buffer 64
echo "~~~MLP + buffer 64~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 64 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_buffer_64 ;

# MLP + buffer 128
echo "~~~MLP + buffer 128~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_buffer_128 ;

# MLP + buffer 256
echo "~~~MLP + buffer 256~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 256 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_buffer_256 ;

# MLP + target_model tm_wait 10
echo "~~~MLP + target_model tm_wait 10~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 1 \
-batch_size 1 \
-n_episodes 1000 \
-target_model \
-tm_wait 10 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_tm_wait_10 ;

# MLP + target_model tm_wait 100
echo "~~~MLP + target_model tm_wait 100~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 1 \
-batch_size 1 \
-n_episodes 1000 \
-target_model \
-tm_wait 100 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_tm_wait_100 ;

# MLP + target_model tm_wait 1000
echo "~~~MLP + target_model tm_wait 1000~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 1 \
-batch_size 1 \
-n_episodes 1000 \
-target_model \
-tm_wait 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_tm_wait_1000 ;

# MLP + buffer + target
echo "~~~MLP + buffer + target~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 256 \
-target_model \
-tm_wait 10 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-run_name MLP_buffer_target ;

# MLP + buffer + target + softmax
echo "~~~MLP + buffer + target + softmax ~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 256 \
-target_model \
-tm_wait 10 \
-n_episodes 1000 \
-gamma 0.99 \
-policy softmax \
-temp 0.1 \
-run_name MLP_buffer_target_softmax ;

# MLP + buffer + target + ucb
echo "~~~MLP + buffer + target + ucb ~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 256 \
-target_model \
-tm_wait 10 \
-n_episodes 1000 \
-gamma 0.99 \
-policy ucb \
-run_name MLP_buffer_target_ucb ;

# MLP + buffer + target + custom reward
echo "~~~MLP + buffer + target + custom rewards ~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 258 \
-target_model \
-tm_wait 10 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-custom_reward \
-run_name MLP_buffer_target_custom-rew ;

# MLP + buffer + target + novelty
echo "~~~MLP + buffer + target + novelty ~~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 258 \
-target_model \
-tm_wait 10 \
-n_episodes 1000 \
-gamma 0.99 \
-policy egreedy \
-epsilon 0.02 0.99 200. \
-intr_rew novelty-based \
-run_name MLP_buffer_target_novelty ;

# MLP + buffer + target + ucb
echo "~~~MLP + buffer + target + ucb~~"
python experiment.py \
-net mlp \
-loss mse \
-optimizer adam \
-optim_lr 1e-3 \
-rb_size 100000 \
-batch_size 256 \
-target_model \
-tm_wait 10 \
-n_episodes 1000 \
-gamma 0.99 \
-policy ucb \
-run_name MLP_buffer_target_ucb ;

# CNN base
echo "~~~CNN base~~~"
python experiment.py \
-use_img \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-virtual_display \
-run_name CNN_base ;

# CNN curiosity
echo "~~~CNN curiosity~~~"
python experiment.py \
-use_img \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew curiosity-based \
-virtual_display \
-run_name CNN_curiosity ;

# CNN novelty
echo "~~~CNN novelty~~~"
python experiment.py \
-use_img \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew novelty-based \
-virtual_display \
-run_name CNN_novelty ;

# CNN dueling
echo "~~~CNN dueling~~~"
python experiment.py \
-use_img \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-dueling \
-virtual_display \
-run_name CNN_dueling ;

# CNN double
echo "~~~CNN double~~~"
python experiment.py \
-use_img \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-double_dql \
-virtual_display \
-run_name CNN_double ;

# CNN double curiosity
echo "~~~CNN double curiosity~~~"
python experiment.py \
-use_img \
-double_dql \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew curiosity-based \
-virtual_display \
-run_name CNN_curiosity_double ;

# CNN dueling curiosity
echo "~~CNN dueling curiosity~~"
python experiment.py \
-use_img \
-dueling \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew curiosity-based \
-virtual_display \
-run_name CNN_curiosity_dueling ;

# CNN double dueling curiosity
echo "~~CNN double dueling curiosity~~"
python experiment.py \
-use_img \
-double_dql \
-dueling \
-net cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew curiosity-based \
-virtual_display \
-run_name CNN_curiosity_double_dueling ;

# SSL base
echo "~~SSL base~~"
python experiment.py \
-use_img \
-ssl_mode 0 \
-net ssl_cnn \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 5 \
-policy egreedy \
-epsilon 0.025 0.99 1000. \
-run_name SSL_base ;

# SSL curiosity
echo "~~SSL with curiosity~~"
python experiment.py \
-use_img \
-net ssl_cnn \
-ssl_mode 0 \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew curiosity-based \
-virtual_display \
-run_name SSL_curiosity ;

# TL base
echo "~~TL base~~"
python experiment.py \
-use_img \
-net tl_cnn \
-tl_mode 0 \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-run_name TL_base ;

# TL curiosity
echo "~~TL with curiosity~~"
python experiment.py \
-use_img \
-net tl_cnn \
-tl_mode 0 \
-loss mse \
-optimizer adam \
-optim_lr 1e-4 \
-rb_size 5000 \
-batch_size 128 \
-n_episodes 1000 \
-gamma 0.99 \
-target_model \
-tm_wait 10 \
-policy egreedy \
-epsilon 0.025 0.99 400. \
-intr_rew curiosity-based \
-run_name TL_curiosity ;