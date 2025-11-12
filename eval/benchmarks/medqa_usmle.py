import os
import re
import random
import pandas as pd
from typing import Optional
from collections import namedtuple
from datasets import load_dataset
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.benchmarks.benchmark import Benchmark

# Example namedtuple matching medmcqa pattern
Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


class MedQAUSMLEBenchmark(Benchmark):
    """MedQA-USMLE-4-options Benchmark.

    A medical question answering benchmark based on USMLE (United States Medical Licensing Examination)
    style questions with 4 answer choices.

    Dataset: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
    """

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def __init__(
        self,
        prompt_type: str = "five_shot",
        system_prompt: str = "You are a helpful assistant with expertise in medicine. Answer the following question to the best of your ability.",
        seed: int = 42,
        num_few_shot_examples: int = 5,
        max_examples: Optional[int] = None,
        name: str = "medqa_usmle"
    ):
        """Initialize MedQA-USMLE benchmark.

        Args:
            prompt_type: Type of prompting ('zero_shot' or 'five_shot')
            system_prompt: System prompt for the model
            seed: Random seed for reproducibility
            num_few_shot_examples: Number of examples for few-shot prompting
            max_examples: Limit number of test examples (for testing)
            name: Benchmark name
        """
        super().__init__(name)

        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.num_few_shot_examples = num_few_shot_examples
        self.max_examples = max_examples
        valid_prompt_types = ['zero_shot', 'five_shot']

        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: {valid_prompt_types}")

        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        """Load MedQA-USMLE dataset from HuggingFace.

        Dataset structure:
            {
                "question": "A 23-year-old pregnant woman at 22 weeks gestation presents...",
                "answer": "Nitrofurantoin",
                "options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
                "meta_info": "step2&3",
                "answer_idx": "D"
            }
        """
        random.seed(self.seed)

        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
        test_data = dataset['test']
        few_shot_examples = dataset['train'].shuffle(seed=self.seed).select(range(self.num_few_shot_examples))

        few_shot_rows = []
        for example in few_shot_examples:
            few_shot_rows.append({
                'question': example['question'].strip(),
                'choice_A': example['options']['A'].strip(),
                'choice_B': example['options']['B'].strip(),
                'choice_C': example['options']['C'].strip(),
                'choice_D': example['options']['D'].strip(),
                'correct_answer': example['answer_idx']
            })
        self.few_shot_df = pd.DataFrame(few_shot_rows)

        rows = []
        for idx, example in enumerate(test_data):
            choices = [
                example['options']['A'],
                example['options']['B'],
                example['options']['C'],
                example['options']['D']
            ]

            correct_answer_idx = self.LETTER_TO_INDEX[example['answer_idx']]
            correct_answer = choices[correct_answer_idx]

            random.shuffle(choices)

            correct_idx = choices.index(correct_answer)
            correct_letter = self.INDEX_TO_LETTER[correct_idx]

            row_data = {
                'id': idx,
                'question': example['question'].strip(),
                'choice_A': choices[0].strip(),
                'choice_B': choices[1].strip(),
                'choice_C': choices[2].strip(),
                'choice_D': choices[3].strip(),
                'correct_answer': correct_letter,
                'correct_answer_text': correct_answer.strip()
            }

            rows.append(row_data)

        self.df = pd.DataFrame(rows)

        if self.max_examples:
            self.df = self.df.sample(n=min(self.max_examples, len(self.df)), random_state=self.seed)

        print(f"Loaded MedQA-USMLE: {len(self.df)} test examples, {len(self.few_shot_df)} few-shot examples")

    def format_user_prompt(self) -> None:
        """Format prompts based on prompt_type."""
        prompts = []

        for _, row in self.df.iterrows():
            example = Example(
                question=row['question'],
                choice1=row['choice_A'],
                choice2=row['choice_B'],
                choice3=row['choice_C'],
                choice4=row['choice_D'],
                correct_index=self.LETTER_TO_INDEX[row['correct_answer']]
            )

            if self.prompt_type == 'zero_shot':
                prompt = self._zero_shot_prompt(example)
            elif self.prompt_type == 'five_shot':
                prompt = self._five_shot_prompt(example)
            else:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

            prompts.append(prompt)

        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def _base_prompt(self, example: Example) -> str:
        """Base prompt format matching medmcqa style."""
        prompt = f"Question: {example.question}\n"
        prompt += f"Choices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        return prompt

    def _zero_shot_prompt(self, example: Example) -> str:
        """Zero-shot prompt - just the question without examples."""
        prompt = f"Format your response as follows: \"The correct answer is (insert answer here)\"\n\n"
        prompt += self._base_prompt(example)
        self.assistant_prompt = "The correct answer is "
        return prompt

    def _five_shot_prompt(self, example: Example) -> str:
        """Five-shot prompt with examples from training set."""
        prompt = "Here are some example questions and answers. Answer the final question yourself. Format your response as follows: \"The correct answer is (insert answer here)\"\n\n"

        for _, row in self.few_shot_df.iterrows():
            prompt += f'Question: {row["question"]}\nChoices:\n'
            prompt += f'(A) {row["choice_A"]}\n'
            prompt += f'(B) {row["choice_B"]}\n'
            prompt += f'(C) {row["choice_C"]}\n'
            prompt += f'(D) {row["choice_D"]}\n'
            prompt += f'The correct answer is ({row["correct_answer"]})\n\n'

        prompt += f"Question: {example.question}"
        prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        self.assistant_prompt = "The correct answer is "

        return prompt

    def score(self) -> None:
        """Parse model responses and compute accuracy scores."""
        scores = []
        parsed_answers = []

        for _, row in self.df.iterrows():
            response = row.get('response', '')
            correct_answer = row['correct_answer']

            parsed = self._parse_answer(response)
            parsed_answers.append(parsed)

            if parsed and parsed == correct_answer:
                scores.append(1)
            else:
                scores.append(0)

        self.df['parsed_answer'] = parsed_answers
        self.df['score'] = scores

        accuracy = sum(scores) / len(scores) if scores else 0.0
        refusal_rate = sum(1 for p in parsed_answers if p is None) / len(parsed_answers)

        print(f"\nMedQA-USMLE Results:")
        print(f"  Overall Accuracy: {accuracy:.2%} ({sum(scores)}/{len(scores)})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")

    def _parse_answer(self, response: str) -> Optional[str]:
        """Extract answer choice (A/B/C/D) from model response.

        Uses the same parsing logic as medmcqa (from GPQA baselines).
        """
        if not response:
            return None

        patterns = [
            r'answer is \((.)\)',
            r'Answer: \((.)\)',
            r'answer: \((.)\)',
            r'answer \((.)\)',
            r'\((.)\)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match and match.group(1) in self.LETTER_TO_INDEX:
                return match.group(1)

        return None

    def save_results(self, output_dir: str) -> None:
        """Save results to parquet file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{self.name}_{self.prompt_type}.parquet"
        )
        self.df.to_parquet(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import pdb; pdb.set_trace()

    benchmark = MedQAUSMLEBenchmark(
        prompt_type="five_shot",
        max_examples=10
    )
    print(benchmark.df['user_prompt'].iloc[0])
    print(benchmark.df['system_prompt'].iloc[0])
    print(benchmark.assistant_prompt)