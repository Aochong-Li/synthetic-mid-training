import os
import sys
import re
import random
import pandas as pd
from typing import Optional, List
import json


# Add LAB-Bench to path to use their utilities
LAB_BENCH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tasks/LAB-Bench'
)
sys.path.insert(0, LAB_BENCH_PATH)

from datasets import load_dataset
from eval.benchmarks.benchmark import Benchmark

from labbench.utils import randomize_choices, ALPHABET, REFUSE_CHOICE


class LABBenchBenchmark(Benchmark):
    """LAB-Bench (Language Agent Biology Benchmark).

    A benchmark for evaluating AI systems on capabilities foundational to
    scientific research in biology, including literature QA, protocol troubleshooting,
    and molecular cloning scenarios.

    Paper: https://arxiv.org/abs/2407.10362
    Dataset: https://huggingface.co/datasets/futurehouse/lab-bench
    """

    VALID_TASKS = ['LitQA2', 'ProtocolQA', 'CloningScenarios']

    def __init__(
        self,
        tasks: str | List[str] = "all",
        use_cot: bool = True,
        system_prompt: str = "You are an expert biology researcher. Think and answer the following question to the best of your ability.",
        seed: int = 42,
        max_examples: Optional[int] = None,
        name: str = "labbench"
    ):
        """Initialize LAB-Bench benchmark.

        Args:
            tasks: Which task(s) to evaluate. Can be:
                   - "all": All tasks (LitQA2, ProtocolQA, CloningScenarios)
                   - Single task name
                   - List of task names
            use_cot: Whether to use chain-of-thought prompting
            system_prompt: System prompt for the model
            seed: Random seed for reproducibility
            max_examples: Limit number of examples (for testing)
            name: Benchmark name
        """
        super().__init__(name)

        # Resolve tasks
        if tasks == "all":
            self.tasks = self.VALID_TASKS
        elif isinstance(tasks, list):
            self.tasks = tasks
        else:
            self.tasks = [tasks]

        self.use_cot = use_cot
        self.system_prompt = system_prompt
        self.seed = seed
        self.max_examples = max_examples

        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        """Load LAB-Bench datasets from HuggingFace using LAB-Bench's exact logic."""
        random.seed(self.seed)

        all_data = []
        for task in self.tasks:
            # Load dataset from HuggingFace
            dataset = load_dataset('futurehouse/lab-bench', task, split='train')

            for example in dataset:
                # Use LAB-Bench's randomize_choices function for exact same logic
                choices, target_output, unsure = randomize_choices(
                    example['ideal'],
                    example['distractors']
                )

                # Prepare question (prepend protocol for ProtocolQA like task.py does)
                question = example['question']
                if task == 'ProtocolQA' and 'protocol' in example:
                    question = example['protocol'] + "\n\n" + question

                row_data = {
                    'id': str(example['id']),
                    'task': task,
                    'subtask': example.get('subtask', task),
                    'question': question,
                    'choices': choices,
                    'correct_answer': target_output,
                    'unsure_answer': unsure,
                    'ideal': example['ideal'],
                    'distractors': example['distractors'],
                    'canary': example['canary'],
                }

                all_data.append(row_data)

        self.df = pd.DataFrame(all_data)

        if self.max_examples:
            self.df = self.df.sample(n=min(self.max_examples, len(self.df)), random_state=self.seed)

        print(f"Loaded LAB-Bench: {len(self.df)} examples from {len(self.tasks)} task(s)")
        for task in self.tasks:
            task_count = len(self.df[self.df['task'] == task])
            print(f"  {task}: {task_count} examples")

    def format_user_prompt(self) -> None:
        """Format prompts using LAB-Bench's exact prompt template."""
        prompts = []

        # Use LAB-Bench's exact prompt template from labbench/zero_shot.py:MCQ_INSTRUCT_TEMPLATE
        cot_prompt = "\nLet's think step by step." if self.use_cot else ""

        for _, row in self.df.iterrows():
            question = row['question']
            choices_str = '\n'.join(row['choices'])

            # This matches labbench/zero_shot.py:MCQ_INSTRUCT_TEMPLATE exactly
            prompt = f"""The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.{cot_prompt}

Question: {question}

Options:
{choices_str}

You MUST include the letter of the correct answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is the correct letter.
Always answer in exactly this format of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

            prompts.append(prompt)

        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def score(self) -> None:
        """Parse model responses and compute accuracy scores using LAB-Bench logic."""
        scores = []
        parsed_answers = []

        for _, row in self.df.iterrows():
            response = row.get('response', '')
            correct_answer = row['correct_answer']
            unsure_answer = row['unsure_answer']

            # Parse answer
            parsed = self._parse_answer(response, row['choices'])
            parsed_answers.append(parsed)

            # Score: correct if parsed answer matches target (LAB-Bench logic)
            if parsed and parsed == correct_answer:
                scores.append(1)
            else:
                scores.append(0)

        self.df['parsed_answer'] = parsed_answers
        self.df['score'] = scores

        # Calculate metrics (following LAB-Bench's Evaluator.compute_metrics)
        n_total = len(scores)
        n_correct = sum(scores)

        # Calculate sure/unsure (coverage) - following LAB-Bench logic
        sure = [p != row['unsure_answer'] if p else False
                for p, (_, row) in zip(parsed_answers, self.df.iterrows())]
        n_sure = sum(sure)

        accuracy = n_correct / n_total if n_total else 0.0
        precision = n_correct / n_sure if n_sure else 0.0
        coverage = n_sure / n_total if n_total else 0.0

        refusal_rate = sum(1 for i, p in enumerate(parsed_answers)
                          if p and p == self.df.iloc[i]['unsure_answer']) / n_total
        failure_rate = sum(1 for p in parsed_answers if p is None) / n_total

        # Store metrics for saving
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'coverage': coverage,
            'refusal_rate': refusal_rate,
            'parse_failure_rate': failure_rate
        }

        print(f"\nLAB-Bench Results:")
        print(f"  Accuracy: {accuracy:.2%} ({n_correct}/{n_total})")
        print(f"  Precision: {precision:.2%}")
        print(f"  Coverage: {coverage:.2%} ({n_sure}/{n_total})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")
        print(f"  Parse Failure Rate: {failure_rate:.2%}")

    def _parse_answer(self, response: str, choices: List[str]) -> Optional[str]:
        """Extract answer from [ANSWER]X[/ANSWER] tags.

        This follows LAB-Bench's parsing logic from zero_shot.py and chembench.
        The actual parsing in LAB-Bench uses regex patterns to extract the letter.
        """
        if not response:
            return None

        # Get valid choice letters from the choices list
        n_choices = len(choices)
        valid_letters = list(ALPHABET[:n_choices])

        # Primary pattern: [ANSWER]X[/ANSWER]
        # This is the main pattern LAB-Bench expects
        match = re.search(r'\[ANSWER\]\s*([A-Z])\s*\[/?ANSWER\]', response, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in valid_letters:
                return answer

        # Fallback: look for any letter in [ANSWER]...[/ANSWER] tags
        match = re.search(r'\[ANSWER\]([^\]]+)\[/?ANSWER\]', response, re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Look for a letter in the content
            for letter in valid_letters:
                if letter in content.upper():
                    return letter

        # Last resort fallback patterns (may not be in official LAB-Bench but useful)
        for pattern in [
            r'answer.*?is\s+\(?([A-Z])\)?',
            r'correct.*?answer.*?\(?([A-Z])\)?',
            r'\(([A-Z])\)',
        ]:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                if answer in valid_letters:
                    return answer

        return None

    def save_results(self, output_dir: str) -> None:
        """Save results to parquet file and metrics to JSON file."""        
        os.makedirs(output_dir, exist_ok=True)

        if len(self.tasks) == 1:
            tasks_part = self.tasks[0]
        elif len(self.tasks) == len(self.VALID_TASKS):
            tasks_part = "all"
        else:
            tasks_part = '_'.join(sorted(self.tasks))

        cot_part = "cot" if self.use_cot else "zero_shot"
        base_filename = f"{self.name}_{tasks_part}_{cot_part}"
        
        # Save parquet file
        parquet_filename = f"{base_filename}.parquet"
        parquet_path = os.path.join(output_dir, parquet_filename)
        self.df.to_parquet(parquet_path, index=False)

        # Save metrics to JSON file
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Results saved to {parquet_path}")
        print(f"Metrics saved to {json_path}")