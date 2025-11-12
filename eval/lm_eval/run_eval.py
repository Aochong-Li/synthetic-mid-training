"""
OpenLMEval: Evaluation framework for language models using vLLM.

Usage:
    python eval/lm_eval/eval.py
    python eval/lm_eval/eval.py model.model_name=meta-llama/Llama-3.1-8B-Instruct
    python eval/lm_eval/eval.py benchmarks.params.subset=diamond
    python eval/lm_eval/eval.py benchmarks_to_run=[gpqa]
"""

import os
import sys
import json
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm_engine import OpenLMEngine, ModelConfig
from eval.benchmarks.benchmark import Benchmark


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    output_dir: str
    metrics: List[str]
    overwrite: bool = False
    enable_thinking: bool = True
    is_base_model: bool = False

class OpenLMEval(OpenLMEngine):
    """Evaluation engine that extends OpenLMEngine with benchmark support."""

    def __init__(
        self,
        model_config: ModelConfig,
        eval_config: EvalConfig,
        benchmarks: Dict[str, Benchmark]
    ):
        if model_config.stop_tokens is None:
            model_config.stop_tokens = ["Question: ", "Choices:\n(A)"]

        super().__init__(model_config)
        self.model_config = model_config
        self.benchmarks = benchmarks
        self.eval_config = eval_config
        self.results = {}

    def format_prompt(self, benchmark: Benchmark) -> Benchmark:
        """Format prompts using chat template."""
        prompts = []
        assistant_prompt = benchmark.assistant_prompt

        for _, row in benchmark.df.iterrows():
            system_prompt = row['system_prompt']
            user_prompt = row["user_prompt"]
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.eval_config.enable_thinking
            )
            if self.eval_config.is_base_model:
                formatted_prompt = formatted_prompt + assistant_prompt

            prompts.append(formatted_prompt)

        benchmark.df["prompt"] = prompts
        return benchmark

    def run_evaluation(self) -> None:
        output_dir = Path(self.eval_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, benchmark in self.benchmarks.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}\n")

            benchmark = self.format_prompt(benchmark)
            responses = self.generate(prompts=benchmark.df["prompt"].tolist())
            
            if self.model_config.n > 1:
                benchmark.df = benchmark.df.loc[
                    np.repeat(benchmark.df.index, self.model_config.n)
                ].reset_index(drop=True)

            benchmark.df = benchmark.df.merge(responses, left_index=True, right_index=True)
            if self.eval_config.is_base_model:
                benchmark.df['response'] = benchmark.df['response'].apply(lambda x: benchmark.assistant_prompt + x)
            benchmark.score()

            os.makedirs(os.path.join(output_dir, name), exist_ok=True)
            benchmark.save_results(os.path.join(output_dir, name))
        
        self.compute_metrics()
        self.save_results(output_dir)

    def compute_metrics(self) -> None:
        self.results = {}

        for name, benchmark in self.benchmarks.items():
            print(f"\n{'='*60}")
            print(f"Results: {name}")
            self.results[name] = {}

            for metric in self.eval_config.metrics:
                if metric == 'mean':
                    value = benchmark.df["score"].mean()
                    self.results[name][metric] = float(value)
                    print(f"  - {metric}: {value:.4f}")
                else:
                    raise ValueError(f"Metric '{metric}' not supported")

            print(f"{'='*60}\n")

    def save_results(self, output_dir: Path = None) -> None:
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"\nAggregate results saved to: {results_path}")

def load_benchmark(benchmark_config: DictConfig) -> Benchmark:
    """Dynamically load a benchmark from config.

    Args:
        benchmark_config: Benchmark configuration from YAML

    Returns:
        Initialized benchmark instance
    """
    module_name = benchmark_config.module
    class_name = benchmark_config.class_name

    module = importlib.import_module(module_name)
    benchmark_class = getattr(module, class_name)

    # Instantiate with parameters
    params = OmegaConf.to_container(benchmark_config.params, resolve=True)
    benchmark = benchmark_class(name=benchmark_config.name, **params)

    return benchmark


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point.

    Args:
        cfg: Hydra configuration object
    """
    print(OmegaConf.to_yaml(cfg))

    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    print(f"Model: {model_config.model_name}")

    eval_config = EvalConfig(**OmegaConf.to_container(cfg.eval, resolve=True))

    print(f"\nLoading benchmarks: {cfg.benchmarks_to_run}")
    benchmarks = {}
    for benchmark_name in cfg.benchmarks_to_run:
        if benchmark_name not in cfg.benchmarks.keys():
            raise ValueError(f"Benchmark '{benchmark_name}' not found in config")
        
        benchmark_cfg = cfg.benchmarks[benchmark_name]
        benchmark = load_benchmark(benchmark_cfg)
        benchmarks[benchmark_name] = benchmark
    
    evaluator = OpenLMEval(
        model_config=model_config,
        eval_config=eval_config,
        benchmarks=benchmarks
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()