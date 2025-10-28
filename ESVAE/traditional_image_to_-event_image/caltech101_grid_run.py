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
                    # [1000, 0.50, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.55, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.60, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.65, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.70, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.75, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.80, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.85, 'CKA', 0.1, 'TCKA'],  # 8-0
                    # [1000, 0.90, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [1000, 0.95, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [1000, 1.00, 'CKA', 0.1, 'TCKA'],  # 8-1
                    #
                    # [2000, 0.50, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [2000, 0.55, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [2000, 0.60, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [2000, 0.65, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [2000, 0.70, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [2000, 0.75, 'CKA', 0.1, 'TCKA'],  # 8-1
                    # [2000, 0.80, 'CKA', 0.1, 'TCKA'],  # 8-2
                    # [2000, 0.85, 'CKA', 0.1, 'TCKA'],  # 8-2
                    # [2000, 0.90, 'CKA', 0.1, 'TCKA'],  # 8-2
                    # [2000, 0.95, 'CKA', 0.1, 'TCKA'],  # 8-2
                    # [2000, 1.00, 'CKA', 0.1, 'TCKA'],  # 8-2
                    #
                    # [3000, 0.50, 'CKA', 0.1, 'TCKA'],  # 8-2
                    # [3000, 0.55, 'CKA', 0.1, 'TCKA'],  # 8-2
                    # [3000, 0.60, 'CKA', 0.1, 'TCKA'],  # 8-2
                    [3000, 0.65, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 0.70, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 0.75, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 0.80, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 0.85, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 0.90, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 0.95, 'CKA', 0.1, 'TCKA'],  # 8-3
                    [3000, 1.00, 'CKA', 0.1, 'TCKA'],  # 8-3
                  ]


for parameter in parameter_list:
    log_name = f"R1_transfer_learning_with_Caltech101-data_set_{parameter[2]}-en_loss_type_{parameter[4]}-fea_loss_type_{parameter[1]}-en_tl_lamb_{parameter[3]}-fea_tl_lamb_{parameter[0]}-seed_1.0-RGB_data_1.0-dvs_data_time_encoder-source_encoder_TET-dvs_encoder_Adam-optim_0.0002-lr"

    os.system(f'CUDA_VISIBLE_DEVICES=\'3\' python tl.py --data_set Caltech101 --lr 0.0002 --optim Adam --epoch 150 '
              f'--seed {parameter[0]} --encoder_tl_lamb {parameter[1]} '
              f'--encoder_tl_loss_type {parameter[2]}  --feature_tl_lamb {parameter[3]} '
              f'--feature_tl_loss_type {parameter[4]} --GPU_id 0 --num_classes 100 2>&1 | tee print_log/caltech101/{log_name}.log')

