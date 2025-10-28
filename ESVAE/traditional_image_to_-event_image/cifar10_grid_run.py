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
                  # [1000, 0.0, 'CKA', 0.2, 'TCKA'],   # 0
                  # [1000, 0.2, 'CKA', 0.0, 'TCKA'],   # 0
                  # [1000, 0.0, 'CKA', 0.1, 'TCKA'],   # 0
                  # [1000, 0.1, 'CKA', 0.0, 'TCKA'],   # 0
                  # [1000, 0.1, 'CKA', 0.1, 'TCKA'],   # 0
                  # [1000, 0.0, 'CKA', 0.2, 'MMD'],    # 0
                  # [1000, 0.0, 'CKA', 0.2, 'MSE'],    # 0
                  [1000, 0.0, 'CKA', 0.2, 'CKA'],    # 0
                  [1000, 0.0, 'CKA', 0.0, 'CKA'],    # 0
                  # [2000, 0.0, 'CKA', 0.2, 'TCKA'],  # 1
                  # [2000, 0.2, 'CKA', 0.0, 'TCKA'],  # 1
                  # [2000, 0.0, 'CKA', 0.1, 'TCKA'],   # 1
                  # [2000, 0.1, 'CKA', 0.0, 'TCKA'],   # 1
                  # [2000, 0.1, 'CKA', 0.1, 'TCKA'],  # 1
                  # [2000, 0.0, 'CKA', 0.2, 'MMD'],  # 1
                  # [2000, 0.0, 'CKA', 0.2, 'MSE'],  # 1
                  # [2000, 0.0, 'CKA', 0.2, 'CKA'],  # 1
                  # [2000, 0.0, 'CKA', 0.0, 'CKA'],  # 1
                  # [3000, 0.0, 'CKA', 0.2, 'TCKA'],   # 2
                  # [3000, 0.2, 'CKA', 0.0, 'TCKA'],   # 2
                  # [3000, 0.0, 'CKA', 0.1, 'TCKA'],   # 2
                  # [3000, 0.1, 'CKA', 0.0, 'TCKA'],   # 2
                  # [3000, 0.1, 'CKA', 0.1, 'TCKA'],   # 2
                  # [3000, 0.0, 'CKA', 0.2, 'MMD'],    # 2
                  # [3000, 0.0, 'CKA', 0.2, 'MSE'],    # 2
                  # [3000, 0.0, 'CKA', 0.2, 'CKA'],    # 2
                  # [3000, 0.0, 'CKA', 0.0, 'CKA'],    # 2
                  ]

for RGB_sample_ratio in [1.0]:  # 0.2, 0.4, 0.6, 0.8, 1.0
    for dvs_sample_ratio in [0.6, 0.8]:
        log_name = (f"R1-transfer_learning_with_CIFAR10-data_set_CKA-en_loss_type_TCKA-fea_loss_type_"
                    f"0.1-en_tl_lamb_0.1-fea_tl_lamb_1000-seed_{RGB_sample_ratio}-RGB_data_{dvs_sample_ratio}-dvs_data_"
                    f"time_encoder-source_encoder_TET-dvs_encoder_Adam-optim_0.0002-lr")

        os.system(f'python tl.py --data_set CIFAR10 --lr 0.0002 --epoch 150 --RGB_sample_ratio {RGB_sample_ratio} '
                  f'--dvs_sample_ratio {dvs_sample_ratio} --optim Adam --GPU_id 1 --num_classes 10 2>&1 | tee print_log/cifar10/{log_name}.log')
