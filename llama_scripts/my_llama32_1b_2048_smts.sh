#!/bin/bash
#SBATCH --job-name=my_llama_3.2_1b_2048_smts_v21_step_bench_neg4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --constraint=a100
#SBATCH --time=48:00:00
#SBATCH --output=logs/my_llama_3.2_1b_2048_smts_v21_step_bench_neg4_%A_%a.out
#SBATCH --error=logs/my_llama_3.2_1b_2048_smts_v21_step_bench_neg4_%A_%a.err
#SBATCH --account conf-icl-2025.09.24-ghanembs



module load cuda/12.4
source /ibex/user/zhuw0b/miniforge/bin/activate /ibex/user/zhuw0b/conda-environments/verl_lora

train_path_1=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/train_data/gsm8k_train_modified.parquet

train_path="[$train_path_1]"



val_path_1=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/eval_data_8/aime24_modified.parquet
val_path_2=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/eval_data_8/aime25_modified.parquet
val_path_3=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/eval_data_8/amc_src_modified.parquet
val_path_4=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/eval_data_8/gsm8k_test_src_modified.parquet
val_path_5=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/eval_data_8/math500_modified.parquet
val_path_6=/ibex/project/c2261/wenxuan/projects/verl_llama/data/my_data_new/step_bench/step_bench_deduplicated.parquet

val_path="[$val_path_1,$val_path_2,$val_path_3,$val_path_4,$val_path_5,$val_path_6]"


PYTHONPATH=.. python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=my_grpo \
    data.train_files=$train_path \
    data.val_files=$val_path \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    +data.enable_thinking=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.importance_sampling_mode="vanilla" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='llama_rl_new' \
    trainer.experiment_name='my_llama_3.2_1b_2048_smts_v21_step_bench_neg4' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1000
