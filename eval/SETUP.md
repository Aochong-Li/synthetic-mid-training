# OpenLMEval Setup Guide

## Overview

OpenLMEval is a modular evaluation framework for language models using vLLM with Hydra configuration management.

## Directory Structure

```
eval/
├── config/                      # Hydra configurations
│   ├── config.yaml             # Main config (ties everything together)
│   ├── eval.yaml               # Evaluation settings
│   ├── model/
│   │   └── default.yaml        # Model configurations
│   └── benchmarks/
│       └── gpqa.yaml           # Benchmark configurations
│
├── lm_eval/
│   └── eval.py                 # Main evaluation script
│
├── benchmarks/
│   ├── benchmark.py            # Abstract benchmark class
│   └── gpqa.py                 # GPQA benchmark implementation
│
├── scripts/
│   ├── run_eval.sh             # Basic evaluation runner
│   ├── run_gpqa_sweep.sh       # Comprehensive GPQA sweep
│   └── README.md               # Scripts documentation
│
└── tasks/
    └── gpqa/                    # Original GPQA repository
        ├── dataset/             # CSV files
        └── prompts/             # Prompt templates
```

## Configuration Files

### 1. Main Config (`config/config.yaml`)

```yaml
defaults:
  - model: default
  - eval: eval
  - benchmarks: gpqa
  - _self_

benchmarks_to_run:
  - gpqa

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: false
```

### 2. Model Config (`config/model/default.yaml`)

```yaml
model_name: meta-llama/Llama-3.1-8B-Instruct
tokenizer_name: null
lora_path: null
max_tokens: 512
temperature: 0.6
n: 1
top_p: 0.95
top_k: 32
max_model_len: 32768
gpu_memory_utilization: 0.75
tensor_parallel_size: 1
# ... more vLLM parameters
```

### 3. Eval Config (`config/eval.yaml`)

```yaml
output_dir: outputs/eval_results
metrics:
  - mean
overwrite: false
enable_thinking: true
batch_size: 32
```

### 4. Benchmark Config (`config/benchmarks/gpqa.yaml`)

```yaml
name: gpqa
class_name: GPQABenchmark
module: eval.benchmarks.gpqa

params:
  subset: diamond
  prompt_type: zero_shot
  system_prompt: "You are a helpful assistant..."
  seed: 42
  max_examples: null
```

## Quick Start

### 1. Install Dependencies

```bash
conda activate rlvr_eval

# Install required packages
pip install hydra-core omegaconf
```

### 2. Extract GPQA Dataset

```bash
cd eval/tasks/gpqa
unzip -P 'deserted-untie-orchid' dataset.zip
cd ../../..
```

### 3. Run Basic Evaluation

```bash
# Using bash script
bash eval/scripts/run_eval.sh

# Or directly with Python
python eval/lm_eval/eval.py
```

### 4. Run with Custom Parameters

```bash
# Override via bash script
bash eval/scripts/run_eval.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --subset diamond \
    --prompt-type chain_of_thought

# Or via Hydra
python eval/lm_eval/eval.py \
    model.model_name=meta-llama/Llama-3.1-8B-Instruct \
    benchmarks.params.subset=diamond \
    benchmarks.params.prompt_type=chain_of_thought
```

## Usage Examples

### Example 1: Evaluate on GPQA Diamond

```bash
python eval/lm_eval/eval.py \
    benchmarks.params.subset=diamond \
    benchmarks.params.prompt_type=zero_shot
```

### Example 2: Chain-of-Thought on Multiple Subsets

```bash
python eval/lm_eval/eval.py \
    benchmarks.params.subset="[diamond,main,extended]" \
    benchmarks.params.prompt_type=chain_of_thought
```

### Example 3: Quick Test with Few Examples

```bash
python eval/lm_eval/eval.py \
    benchmarks.params.max_examples=10 \
    eval.output_dir=outputs/test
```

### Example 4: Large Model with Model Parallelism

```bash
python eval/lm_eval/eval.py \
    model.model_name=meta-llama/Llama-3.1-70B-Instruct \
    model.tensor_parallel_size=4 \
    model.gpu_memory_utilization=0.9
```

### Example 5: Comprehensive Sweep

```bash
bash eval/scripts/run_gpqa_sweep.sh meta-llama/Llama-3.1-8B-Instruct
```

## How It Works

### 1. Configuration Loading

```python
# Hydra loads and merges configs
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # cfg contains merged configuration from all YAML files
    pass
```

### 2. Benchmark Initialization

```python
# Dynamically load benchmark class from config
module = importlib.import_module(cfg.benchmarks.module)
benchmark_class = getattr(module, cfg.benchmarks.class_name)
benchmark = benchmark_class(**cfg.benchmarks.params)
```

### 3. Data Loading

```python
# Load dataset and format prompts
benchmark.load_dataset()      # Loads CSV, shuffles choices
benchmark.format_user_prompt()  # Creates prompts using templates
```

### 4. Evaluation Pipeline

```python
evaluator = OpenLMEval(model_config, eval_config, benchmarks)

# For each benchmark:
#   1. Format prompts with chat template
#   2. Generate responses using vLLM
#   3. Parse answers from responses
#   4. Compute accuracy
#   5. Save results
evaluator.run_evaluation()
```

## Hydra Override Syntax

Hydra uses dot notation to override nested configurations:

```bash
# Override model parameters
python eval/lm_eval/eval.py model.model_name=llama-8b

# Override benchmark parameters
python eval/lm_eval/eval.py benchmarks.params.subset=diamond

# Override eval parameters
python eval/lm_eval/eval.py eval.output_dir=custom_outputs

# Multiple overrides
python eval/lm_eval/eval.py \
    model.temperature=0.7 \
    model.max_tokens=1024 \
    benchmarks.params.prompt_type=chain_of_thought \
    eval.enable_thinking=false
```

## Adding New Benchmarks

### 1. Implement Benchmark Class

Create `eval/benchmarks/my_benchmark.py`:

```python
from eval.benchmarks.benchmark import Benchmark

class MyBenchmark(Benchmark):
    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        # Store parameters

    def load_dataset(self) -> None:
        # Load data into self.df
        pass

    def format_user_prompt(self) -> None:
        # Add 'system_prompt' and 'user_prompt' columns to self.df
        pass

    def score(self) -> None:
        # Parse responses and add 'score' column
        pass

    def save_results(self, output_dir: str) -> None:
        # Save results
        pass
```

### 2. Create Config File

Create `eval/config/benchmarks/my_benchmark.yaml`:

```yaml
name: my_benchmark
class_name: MyBenchmark
module: eval.benchmarks.my_benchmark

params:
  # Your benchmark parameters
  param1: value1
  param2: value2
```

### 3. Use It

```bash
python eval/lm_eval/eval.py \
    benchmarks=my_benchmark \
    benchmarks_to_run=[my_benchmark]
```

## Output Files

### Aggregate Results (`results.json`)

```json
{
  "gpqa": {
    "mean": 0.4567
  }
}
```

### Detailed Results (`gpqa_diamond_zero_shot.parquet`)

Columns:
- `id`: Question ID
- `question`: Question text
- `choice_A`, `choice_B`, `choice_C`, `choice_D`: Answer choices
- `correct_answer`: Correct answer letter (A/B/C/D)
- `gpqa_diamond`, `gpqa_main`, `gpqa_extended`: Subset membership
- `system_prompt`, `user_prompt`: Prompts
- `prompt`: Formatted prompt with chat template
- `response`: Model response
- `parsed_answer`: Extracted answer letter
- `score`: 1 if correct, 0 otherwise

## Key Features

### ✅ Modular Design
- Separate configs for model, eval, and benchmarks
- Easy to add new benchmarks
- Override any parameter via command line

### ✅ Multi-Subset Support
- Load single or multiple GPQA subsets
- Automatic deduplication
- Track subset membership

### ✅ Flexible Prompting
- Zero-shot, few-shot, chain-of-thought
- Custom system prompts
- Template-based formatting

### ✅ Production-Ready
- vLLM for fast inference
- Model parallelism support
- Result caching and saving

### ✅ Reproducible
- Seed control for random shuffling
- Config versioning via Hydra
- All parameters logged

## Troubleshooting

### Hydra Config Not Found
```
Error: Could not find config file
```
**Solution:** Run from project root directory

### Import Errors
```
ModuleNotFoundError: No module named 'eval'
```
**Solution:** Set PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Out of Memory
```
OutOfMemoryError: ...
```
**Solution:** Reduce batch size or increase parallelism
```bash
python eval/lm_eval/eval.py \
    model.tensor_parallel_size=2 \
    model.gpu_memory_utilization=0.8
```

### Dataset Not Found
```
FileNotFoundError: Dataset not found at eval/tasks/gpqa/dataset/gpqa_diamond.csv
```
**Solution:** Extract dataset
```bash
unzip -P 'deserted-untie-orchid' eval/tasks/gpqa/dataset.zip -d eval/tasks/gpqa/
```

## Next Steps

1. **Add more benchmarks**: Follow the pattern in `eval/benchmarks/gpqa.py`
2. **Custom metrics**: Extend `compute_metrics()` in `eval.py`
3. **Distributed evaluation**: Use Ray with `model.data_parallel_replicas`
4. **Result analysis**: Use pandas to analyze `.parquet` files

## References

- [Hydra Documentation](https://hydra.cc/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [GPQA Paper](https://arxiv.org/abs/2311.12022)
