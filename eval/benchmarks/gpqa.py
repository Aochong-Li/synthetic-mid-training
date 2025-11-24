import os
import re
import json
import random
import pandas as pd
from typing import Optional, Union, List
from collections import namedtuple
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.benchmarks.benchmark import Benchmark

# Example namedtuple from GPQA
Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


class GPQABenchmark(Benchmark):
    """GPQA (Graduate-Level Google-Proof Q&A) Benchmark.

    Evaluates models on graduate-level science questions across physics,
    chemistry, and biology. Uses the same prompt templates as the original
    GPQA repository.
    """

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def __init__(
        self,
        subset: Union[str, List[str]] = "diamond",
        prompt_type: str = "zero_shot",
        system_prompt: str = "You are a helpful assistant with expertise in science. Answer the following question to the best of your ability.",
        seed: int = 42,
        max_examples: Optional[int] = None,
        name: str = "gpqa"
    ):
        """Initialize GPQA benchmark.

        Args:
            subset: Which GPQA subset(s) to load. Can be a string or list of strings.
                   Options: 'main', 'diamond', 'extended', 'experts'
                   If list, loads and deduplicates across subsets.
            prompt_type: Prompting strategy ('zero_shot', 'chain_of_thought', '5_shot')
            seed: Random seed for answer shuffling
            max_examples: Limit number of examples (for testing)
            name: Benchmark name
        """
        super().__init__(name)

        self.subset = [subset] if isinstance(subset, str) else subset
        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.max_examples = max_examples

        tasks_dir = os.path.join(os.path.dirname(__file__), '..', 'tasks', 'gpqa')
        self.data_dir = os.path.join(tasks_dir, 'dataset')
        self.prompts_dir = os.path.join(tasks_dir, 'prompts')

        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        """Load GPQA dataset from CSV(s) and shuffle answer choices.

        Loads one or more subsets, deduplicates across subsets, and tracks
        which subset(s) each question belongs to.
        """
        random.seed(self.seed)

        all_subsets = []
        for subset_name in ['main', 'diamond', 'extended']:
            csv_path = os.path.join(self.data_dir, f"gpqa_{subset_name}.csv")

            if not os.path.exists(csv_path):
                question_df = load_dataset("Idavidrein/gpqa", 'gpqa_' + subset_name)['train'].to_pandas()
            else:
                question_df = pd.read_csv(csv_path)

            question_df['_source_subset'] = subset_name
            all_subsets.append(question_df)
            
        all_subsets = pd.concat(all_subsets, ignore_index=True)
        all_subsets['_question_key'] = all_subsets['Question'].str.strip() + '||' + all_subsets['Correct Answer'].str.strip()
        subset_membership = all_subsets.groupby('_question_key')['_source_subset'].apply(set).to_dict()
        
        deduplicated_df = all_subsets[all_subsets['_source_subset'].isin(self.subset)].drop_duplicates(subset='_question_key', keep='first')

        rows = []
        for idx, row in deduplicated_df.iterrows():
            question_key = row['_question_key']
            subsets_for_question = subset_membership[question_key]

            choices = [
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3'],
                row['Correct Answer']
            ]
            random.shuffle(choices)

            correct_idx = choices.index(row['Correct Answer'])
            correct_letter = self.INDEX_TO_LETTER[correct_idx]

            row_data = {
                'id': idx,
                'record_id': row['Record ID'],
                'question': row['Question'].strip(),
                'choice_A': choices[0].strip(),
                'choice_B': choices[1].strip(),
                'choice_C': choices[2].strip(),
                'choice_D': choices[3].strip(),
                'correct_answer': correct_letter,
                'correct_answer_text': row['Correct Answer'].strip(),
                'subdomain': row.get('Subdomain', ''),
                'gpqa_diamond': 'diamond' in subsets_for_question,
                'gpqa_main': 'main' in subsets_for_question,
                'gpqa_extended': 'extended' in subsets_for_question,
            }

            rows.append(row_data)

        self.df = pd.DataFrame(rows)

        if self.max_examples:
            self.df = self.df.head(self.max_examples)

        subset_str = ', '.join(self.subset)
        print(f"Final dataset: {len(self.df)} examples from GPQA subsets [{subset_str}]")
        print(f"  - In diamond: {self.df['gpqa_diamond'].sum()}")
        print(f"  - In main: {self.df['gpqa_main'].sum()}")
        print(f"  - In extended: {self.df['gpqa_extended'].sum()}")

    def format_user_prompt(self) -> None:
        """Format prompts using GPQA's prompt templates."""
        prompts = []

        for _, row in self.df.iterrows():
            # Create Example namedtuple
            example = Example(
                question=row['question'],
                choice1=row['choice_A'],
                choice2=row['choice_B'],
                choice3=row['choice_C'],
                choice4=row['choice_D'],
                correct_index=self.LETTER_TO_INDEX[row['correct_answer']]
            )

            # Format prompt based on type
            if self.prompt_type == 'zero_shot':
                prompt = self._zero_shot_prompt(example)
            elif self.prompt_type == 'chain_of_thought':
                prompt = self._chain_of_thought_prompt(example)
            elif self.prompt_type == '5_shot':
                raise ValueError(f"5-shot prompt is not supported")
            else:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

            prompts.append(prompt)

        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def _base_prompt(self, example: Example) -> str:
        """Base prompt format from GPQA."""
        prompt = f"What is the correct answer to this question: {example.question}"
        prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        return prompt

    def _zero_shot_prompt(self, example: Example) -> str:
        """Zero-shot prompt (from GPQA utils.py)."""
        prompt = f"Format your response as follows: \"The correct answer is (insert answer here)\".\n\n"
        prompt += self._base_prompt(example)
        
        self.assistant_prompt = "The correct answer is "

        return prompt

    def _chain_of_thought_prompt(self, example: Example) -> str:
        """Chain-of-thought prompt with examples (from GPQA utils.py)."""
        json_path = os.path.join(self.prompts_dir, 'chain_of_thought_examples.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        prompt = "For the question below, think step by step and generate your response as follows: \"The correct answer is (insert answer here)\".\n"
        for q in json_data["questions"]:
            prompt += f'\nQuestion: {q["question"]}\nChoices:\n'
            for choice, value in q["choices"].items():
                prompt += f'({choice}) {value}\n'
            prompt += f"Let's think step by step:\n{q['explanation']}\n"
            prompt += f'The correct answer is ({q["correct_answer"]})\n'
        prompt += f"\nQuestion: {example.question}"
        prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}\n"
        
        # prompt += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
        # prompt += "\nLet's think step by step:\n"

        self.assistant_prompt = "Let's think step by step:\n"

        return prompt

    def _five_shot_prompt(self, example: Example) -> str:
        """5-shot prompt without explanations (from GPQA utils.py)."""
        prompt = "Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.\n"

        # Load few-shot examples
        json_path = os.path.join(self.prompts_dir, 'chain_of_thought_examples.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Add examples without explanations
        for q in json_data["questions"]:
            prompt += f'Question: {q["question"]}\nChoices:\n'
            for choice, value in q["choices"].items():
                prompt += f'({choice}) {value}\n'
            prompt += f'The correct answer is ({q["correct_answer"]})\n'

        # Add current question
        prompt += f"Question: {example.question}"
        prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        prompt += "\nWhen you're ready to answer, please use the format \"The correct answer is (insert answer here).\"\n"

        return prompt

    def score(self) -> None:
        """Parse model responses and compute accuracy scores."""
        scores = []
        parsed_answers = []

        for _, row in self.df.iterrows():
            response = row.get('response', '')
            correct_answer = row['correct_answer']

            # Parse answer from response
            parsed = self._parse_answer(response)
            parsed_answers.append(parsed)

            # Score: 1 if correct, 0 otherwise
            if parsed and parsed == correct_answer:
                scores.append(1)
            else:
                scores.append(0)

        self.df['parsed_answer'] = parsed_answers
        self.df['score'] = scores

        # Print summary
        accuracy = sum(scores) / len(scores) if scores else 0.0
        refusal_rate = sum(1 for p in parsed_answers if p is None) / len(parsed_answers)
        subset_str = ', '.join(self.subset)
        print(f"\nGPQA [{subset_str}] Results:")
        print(f"  Accuracy: {accuracy:.2%} ({sum(scores)}/{len(scores)})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")

    def _parse_answer(self, response: str) -> Optional[str]:
        """Extract answer choice (A/B/C/D) from model response.

        Uses the same parsing logic as GPQA's parse_sampled_answer.
        """
        if not response:
            return None

        # Try multiple patterns (from GPQA baselines/run_baseline.py)
        patterns = [
            r'answer is \(([A-J])\)',
            r'Answer: \(([A-J])\)',
            r'answer: \(([A-J])\)',
            r'answer \(([A-J])\)',
            r'\(([A-J])\)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match and match.group(1) in self.LETTER_TO_INDEX:
                return match.group(1)

        return None

    def save_results(self, output_dir: str) -> None:
        """Save results to parquet file."""
        os.makedirs(output_dir, exist_ok=True)
        # Join subset names with underscore
        subset_str = '_'.join(sorted(self.subset))
        output_path = os.path.join(
            output_dir,
            f"{self.name}_{subset_str}_{self.prompt_type}.parquet"
        )
        self.df.to_parquet(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    
    benchmark = GPQABenchmark(
        subset="diamond",
        prompt_type="chain_of_thought",
        max_examples=10
    )

    print(benchmark.df['user_prompt'][0])
    print(benchmark.df['system_prompt'][0])
    print(benchmark.assistant_prompt)