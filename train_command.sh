#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard}"

cd "${PROJECT_ROOT}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    source activate "${CONDA_ENV}"
fi

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${MODEL_DIR}/Qwen3-14B}"
n_nodes="${N_NODES:-4}"
n_gpus_per_node="${N_GPUS_PER_NODE:-8}"
tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
save_freq="${SAVE_FREQ:-500}"

dapo_train_path="${TRAIN_FILE:-${DATA_DIR}/deepscaler.parquet}"
r1_test_path="${VAL_FILE:-${DATA_DIR}/validation.parquet}"

experiment_name="${EXPERIMENT_NAME:-anonymous}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-8192}"
output_dir="${OUTPUT_DIR:-${CKPT_DIR}/${experiment_name}}"
log_dir="${RUN_LOG_DIR:-${LOG_DIR}/${experiment_name}}"
mkdir -p "${log_dir}" "${TENSORBOARD_DIR}"
log_file="${log_dir}/train.log"


exec > >(tee -a "${log_file}") 2>&1

set -x
rollout_n="${ROLLOUT_N:-4}"
use_token_entropy_weighting="${USE_TOKEN_ENTROPY_WEIGHTING:-true}"
baseline="${USE_BASELINE_ZERO_WEIGHT:-true}"
use_hybrid_exploration="${USE_HYBRID_EXPLORATION:-true}"
enable_thinking_mode="${ENABLE_THINKING_MODE:-true}"

t1="${T1:-1.2}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.per.enable_token_entropy_weighting=${use_token_entropy_weighting} \
    algorithm.per.warmup_steps=20 \
    algorithm.per.entropy_t0=0.3 \
    algorithm.per.entropy_t1=${t1} \
    actor_rollout_ref.rollout.hybrid_exploration.enable=${use_hybrid_exploration} \
    actor_rollout_ref.rollout.hybrid_exploration.use_baseline_zero_weight=${baseline} \
    actor_rollout_ref.rollout.hybrid_exploration.temperatures=[0.3,${t1}] \
    actor_rollout_ref.rollout.hybrid_exploration.top_k_values=[-1,-1] \
    data.train_files=${dapo_train_path} \
    data.val_files=${r1_test_path} \
    data.train_batch_size=128 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.truncation=left \
    +data.apply_chat_template_kwargs.enable_thinking=${enable_thinking_mode} \
    actor_rollout_ref.model.path=${PRETRAINED_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=anonymous_review \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${n_nodes} \
    trainer.resume_mode=auto \
    trainer.save_freq=${save_freq} \
    trainer.val_before_train=False \
    trainer.default_local_dir=${output_dir} \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    "$@"
