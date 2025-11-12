#!/bin/bash

set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define model configurations
models=(
    # "Qwen/Qwen3-1.7B:./outputs/qwen3-1.7b:False"
    "Qwen/Qwen3-1.7B-Base:./outputs/qwen3-1.7b-base:True"
    "Qwen/Qwen3-4B:./outputs/qwen3-4b:False"
    "Qwen/Qwen3-4B-Base:./outputs/qwen3-4b-base:True"
)

# Loop through each model configuration
for model_config in "${models[@]}"; do
    IFS=':' read -r model_name output_dir is_base_model <<< "$model_config"
    
    python -m eval.lm_eval.run_eval \
        model.model_name="$model_name" \
        model.max_tokens=32768 \
        model.temperature=0.6 \
        model.n=4 \
        model.top_p=0.95 \
        model.top_k=32 \
        model.gpu_memory_utilization=0.9 \
        model.dtype=bfloat16 \
        model.max_num_batched_tokens=32768 \
        model.tensor_parallel_size=4 \
        model.data_parallel_replicas=1 \
        eval.output_dir="$output_dir" \
        eval.is_base_model=$is_base_model
done