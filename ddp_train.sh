#!/bin/bash

### 環境変数
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
export MASTER_ADDR=localhost
export MASTER_PORT=12345   
export NODE_RANK=0
export NUM_NODES=1
export NUM_GPUS_PER_NODE=15
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export XTYPE='audio'
export CTYPE='text'
#export CUDA_LAUNCH_BLOCKING=1


### スクリプト実行
#python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK train.py
#python train_dp.py
#torchrun  train2.py
python train_pl.py