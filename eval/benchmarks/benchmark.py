import os
import pandas as pd
from abc import ABC, abstractmethod


class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        self.df = pd.DataFrame()
        self.assistant_prompt = ""

    @abstractmethod
    def load_dataset(self) -> None:
        pass
    
    @abstractmethod
    def format_user_prompt(self) -> None:
        pass

    @abstractmethod
    def score(self) -> None:
        pass

    @abstractmethod
    def save_results(self, output_dir: str) -> None:
        self.df.to_parquet(os.path.join(output_dir, f"{self.name}.parquet"))