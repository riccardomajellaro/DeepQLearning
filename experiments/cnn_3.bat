python experiment.py ^
-use_img ^
-net cnn ^
-double_dql ^
-loss mse ^
-optimizer rms ^
-optim_lr 1e-4 ^
-rb_size 5000 ^
-batch_size 128 ^
-n_episodes 5000 ^
-gamma 0.9 ^
-target_model ^
-tm_wait 250 ^
-custom_reward ^
-policy egreedy ^
-epsilon 0.05 0.99 1000. ^
-run_name C:/Users/ricca/Desktop/exp_results/cnn_3