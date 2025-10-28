# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: caltech101_grid_run.py.py
@time: 2022/9/5 11:34
"""

import os

# seed, encoder_tl_lamb, encoder_tl_loss_type, feature_tl_lamb, feature_tl_loss_type
parameter_list = [
                  # [2000, 0.0, 'CKA', 0.1, 'TCKA'],  # 2-2
                  # [2000, 0.1, 'CKA', 0.0, 'TCKA'],  # 2-2
                  # [2000, 0.0, 'CKA', 0.2, 'MMD'],   # 2-2
                  # [2000, 0.0, 'CKA', 0.2, 'MSE'],   # 2-2
                  # [2000, 0.0, 'CKA', 0.2, 'CKA'],   # 2-2
                  [2000, 0.0, 'CKA', 0.0, 'CKA'],   # 4-0
                  # [3000, 0.0, 'CKA', 0.2, 'TCKA'],  # 4-0
                  # [3000, 0.2, 'CKA', 0.0, 'TCKA'],  # 4-0
                  # [3000, 0.0, 'CKA', 0.1, 'TCKA'],  # 4-0
                  # [3000, 0.1, 'CKA', 0.0, 'TCKA'],  # 4-0
                  # [3000, 0.1, 'CKA', 0.1, 'TCKA'],  # 4-1
                  # [3000, 0.0, 'CKA', 0.2, 'MMD'],   # 4-1
                  # [3000, 0.0, 'CKA', 0.2, 'MSE'],   # 4-1
                  # [3000, 0.0, 'CKA', 0.2, 'CKA'],   # 4-1
                  [3000, 0.0, 'CKA', 0.0, 'CKA'],  # 4-0
                  ]


for parameter in parameter_list:
    log_name = f"transfer_learning_with_MNIST-data_set_{parameter[2]}-en_loss_type_{parameter[4]}-fea_loss_type_{parameter[1]}-en_tl_lamb_{parameter[3]}-fea_tl_lamb_{parameter[0]}-seed_1.0-RGB_data_1.0-dvs_data_time_encoder-source_encoder_TET-dvs_encoder_Adam-optim_0.0001-lr"

    os.system(f'python tl.py --data_set MNIST --lr 0.0001 --epoch 80 '
              f'--seed {parameter[0]} --encoder_tl_lamb {parameter[1]} '
              f'--encoder_tl_loss_type {parameter[2]}  --feature_tl_lamb {parameter[3]} '
              f'--feature_tl_loss_type {parameter[4]} --GPU_id 0 2>&1 | tee print_log/mnist/{log_name}.log')
