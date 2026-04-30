set -x

nproc_per_node=1
save_path=./models/sft

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME${DATA_DIR}/train.parquet \
    data.val_files=$HOME${DATA_DIR}/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=256 \
    data.train_batch_size=256 \
    use_remove_padding=True \
    model.partial_pretrain=HuggingFaceTB/SmolLM2-135M-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=char_count-sft \
    trainer.experiment_name=char_count-sft-SmolLM2-135M-Instruct \
    trainer.total_epochs=3 \
    trainer.logger=console