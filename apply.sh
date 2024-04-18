#!/bin/bash

python train.py --problem original --data_dir ./test.json --image_size 512 --n_level 7 --depth 32 --flow_permutation 3 --flow_coupling 1 --seed 0 --lr 0.001 --n_bits_x 8 --n_batch_train 128 --epochs_full_valid 5 --epochs_full_sample 5 --inference --restore_path ./models/model_best_loss.ckpt --logdir ./logs_test --q_path ./models/q.npy --ratioOriginalZ 0.2 --q_Linf --q_amplifier 1.2 --image_color_num 1


