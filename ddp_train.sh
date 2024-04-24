#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,4,5,6,7,8,9,10,11,12,13,14,15
export MASTER_ADDR=localhost
export MASTER_PORT=12345   
export NODE_RANK=0
export NUM_NODES=1
export NUM_GPUS_PER_NODE=13
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export XTYPE='audio'
export CTYPE='text'

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK train.py