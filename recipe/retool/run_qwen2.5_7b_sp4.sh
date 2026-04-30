#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

set -x

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1

ulimit -n 65535

EXPERIMENT_NAME=retool-multiturn-sft-qwen2.5-7b-sp4

torchrun --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.max_length=16384 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=16 \
    data.train_files=$HOME${DATA_DIR}/train.parquet \
    data.val_files=$HOME${DATA_DIR}/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    model.partial_pretrain=$HOME/models/Qwen/Qwen2.5-7B-Instruct \
    model.trust_remote_code=true \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    optim.lr=1e-6 \
    trainer.default_local_dir=$HOME/checkpoints/retool-multiturn-sft/$EXPERIMENT_NAME \
    trainer.project_name=retool-multiturn-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=8 $@ \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
