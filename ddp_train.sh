#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export MASTER_ADDR=localhost
export MASTER_PORT=12345   
export NODE_RANK=0
export NUM_NODES=1
export NUM_GPUS_PER_NODE=5
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK train.py