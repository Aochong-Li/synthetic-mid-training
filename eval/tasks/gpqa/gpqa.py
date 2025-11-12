from benchmark import Benchmark
from core.llm_engine import ModelConfig
from datasets import load_dataset
import pandas as pd

from typing import Union, List

class GPQA(Benchmark):
    def __init__(
        self,
        config: ModelConfig,
        subsets: Union[str, List[str]] = "diamond",
        prompt_type: str = "zero_shot",
        benchmark_name: str = "gpqa"
        ):
        super().__init__(config, benchmark_name)
        self.subset = subsets
        self.prompt_type = prompt_type

    def load_dataset(self) -> None:
        self.df = []
        subsets = [self.subsets] if isinstance(self.subsets, str) else self.subsets
        
        for subset in subsets:
            dataset = load_dataset("Idavidrein/gpqa", self.subset)['train'].to_pandas()
            columns = ['Record ID', 'Subdomain', 'Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
            subset_df = dataset[columns].rename(columns={'Record ID': 'id'})
            subset_df['split'] = subset
            self.df.append(subset_df)
        self.df = pd.concat(self.df, ignore_index=True)
    
    