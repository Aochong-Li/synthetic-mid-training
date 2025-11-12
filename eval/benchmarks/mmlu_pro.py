import os
import re
import random
import pandas as pd
from typing import Optional, List, Union
from datasets import load_dataset
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.benchmarks.benchmark import Benchmark

def extract_answer_mmlu_pro(text):
    """Level 1 extraction: Basic pattern matching."""
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


def extract_answer_again(text):
    """Level 2 fallback: Alternative answer format."""
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_answer_final(text)


def extract_answer_final(text):
    """Level 3 fallback: Last occurrence of single letter."""
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


class MMLUProBenchmark(Benchmark):
    """MMLU-Pro Benchmark.

    An enhanced version of MMLU with 10 options per question and more challenging content.

    Paper: https://arxiv.org/abs/2406.01574
    Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """

    CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    LETTER_TO_INDEX = {letter: i for i, letter in enumerate(CHOICES)}
    INDEX_TO_LETTER = {i: letter for i, letter in enumerate(CHOICES)}
    STEM_CATEGORIES = ["math", "physics", "chemistry", "biology", "computer science", "engineering", 'health', "other"]

    def __init__(
        self,
        prompt_type: str = "cot",
        system_prompt: str = "You are a helpful assistant.",
        seed: int = 42,
        num_few_shot_examples: int = 5,
        max_examples: Optional[int] = None,
        categories: Union[List[str], str] = "stem",
        name: str = "mmlu_pro"
    ):
        """Initialize MMLU-Pro benchmark.

        Args:
            prompt_type: Type of prompting ('zero_shot' or 'cot')
            system_prompt: System prompt for the model
            seed: Random seed for reproducibility
            num_few_shot_examples: Number of examples for CoT prompting
            max_examples: Limit number of test examples (for testing)
            categories: List of categories to evaluate (None = all categories)
            name: Benchmark name
        """
        super().__init__(name)

        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.num_few_shot_examples = num_few_shot_examples
        self.max_examples = max_examples

        if categories == "stem":
            self.categories = self.STEM_CATEGORIES
        else:
            self.categories = None if categories == "all" or categories == ["all"] else categories

        valid_prompt_types = ['zero_shot', 'cot']
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: {valid_prompt_types}")

        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        """Load MMLU-Pro dataset from HuggingFace.

        Dataset structure:
            {
                "question_id": 70,
                "question": "Typical advertising regulatory bodies suggest...",
                "options": ["Safe practices, Fear, Jealousy, Trivial", ...],
                "answer": "I",
                "answer_index": 8,
                "cot_content": "",
                "category": "business",
                "src": "ori_mmlu-business_ethics"
            }
        """
        random.seed(self.seed)

        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        test_data = dataset['test']
        val_data = dataset['validation']

        # Preprocess: remove "N/A" options (from original preprocess function)
        test_data = self._preprocess(test_data)
        val_data = self._preprocess(val_data)

        self.val_by_category = {}
        for example in val_data:
            cat = example['category']
            if cat not in self.val_by_category:
                self.val_by_category[cat] = []
            self.val_by_category[cat].append(example)

        # Process test data
        rows = []
        for example in test_data:
            if self.categories is not None and example['category'] not in self.categories:
                continue

            options = example['options']
            num_options = len(options)
            answer_letter = example['answer']
            answer_index = example['answer_index']

            row_data = {
                'question_id': example['question_id'],
                'question': example['question'].strip(),
                'num_options': num_options,
                'correct_answer': answer_letter,
                'correct_answer_index': answer_index,
                'category': example['category'],
                'src': example['src'],
            }

            for i, option in enumerate(options):
                row_data[f'choice_{self.CHOICES[i]}'] = option.strip()

            rows.append(row_data)

        self.df = pd.DataFrame(rows)

        if self.max_examples:
            self.df = self.df.sample(n=min(self.max_examples, len(self.df)), random_state=self.seed)

        print(f"Loaded MMLU-Pro: {len(self.df)} test examples")
        if self.categories is not None:
            print(f"  Filtered by categories: {self.categories}")

    def _preprocess(self, data):
        """Remove 'N/A' options from dataset.

        This matches the preprocess() function in evaluate_from_local.py lines 40-50.
        """
        res = []
        for each in data:
            options = [opt for opt in each['options'] if opt != "N/A"]
            each['options'] = options
            res.append(each)
        return res

    def format_user_prompt(self) -> None:
        """Format prompts based on prompt_type."""
        prompts = []

        for _, row in self.df.iterrows():
            if self.prompt_type == 'zero_shot':
                prompt = self._zero_shot_prompt(row)
            elif self.prompt_type == 'cot':
                prompt = self._cot_prompt(row)
            else:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

            prompts.append(prompt)

        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def _format_question_with_options(self, question: str, options: List[str]) -> str:
        """Format question with options using A., B., C., etc."""
        formatted = f"Question: {question}\n"
        formatted += "Options:\n"
        for i, option in enumerate(options):
            formatted += f"{self.CHOICES[i]}. {option}\n"
        return formatted

    def _zero_shot_prompt(self, row) -> str:
        """Zero-shot prompt - just the question without examples."""
        num_options = row['num_options']
        options = [row[f'choice_{self.CHOICES[i]}'] for i in range(num_options)]

        category = row['category']

        prompt = f"The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n\n"
        prompt += self._format_question_with_options(row['question'], options)
        prompt += "Thinking: Let's think step by step."
        self.assistant_prompt = "The answer is "

        return prompt

    def _cot_prompt(self, row) -> str:
        """Chain-of-thought prompt with few-shot examples from validation set."""
        category = row['category']

        # Initial prompt matching MMLU-Pro format
        prompt = f"The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n\n"
        val_examples = self.val_by_category[category]
        k = min(self.num_few_shot_examples, len(val_examples))
        sampled_examples = val_examples[:k]

        for example in sampled_examples:
            ex_options = example['options']
            prompt += self._format_question_with_options(example['question'], ex_options)

            cot_content = example.get('cot_content', '')
            if cot_content:
                cot_content = cot_content.replace("A: Let's think step by step.",
                                                 "Thinking: Let's think step by step.")
                prompt += cot_content + "\n\n"
            else:
                prompt += f"Thinking: The answer is ({example['answer']})\n\n"

        num_options = row['num_options']
        options = [row[f'choice_{self.CHOICES[i]}'] for i in range(num_options)]
        prompt += self._format_question_with_options(row['question'], options)
        # prompt += "Thinking: Let's think step by step."
        self.assistant_prompt = "Thinking: Let's think step by step."

        return prompt

    def score(self) -> None:
        """Parse model responses and compute accuracy scores using MMLU-Pro's exact logic."""
        scores = []
        parsed_answers = []

        for _, row in self.df.iterrows():
            response = row.get('response', '')
            correct_answer = row['correct_answer']

            parsed = self._parse_answer(response)
            parsed_answers.append(parsed)

            # Score: 1 if correct, 0 otherwise
            if parsed and parsed == correct_answer:
                scores.append(1)
            else:
                scores.append(0)

        self.df['parsed_answer'] = parsed_answers
        self.df['score'] = scores

        accuracy = sum(scores) / len(scores) if scores else 0.0
        refusal_rate = sum(1 for p in parsed_answers if p is None) / len(parsed_answers)

        print(f"\nMMLU-Pro Results:")
        print(f"  Overall Accuracy: {accuracy:.2%} ({sum(scores)}/{len(scores)})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")

    def _parse_answer(self, response: str) -> Optional[str]:
        """Extract answer using MMLU-Pro's exact extraction logic.

        Uses level 'l2' extraction which includes fallback patterns:
        1. Try "answer is (X)" pattern
        2. Try "Answer: X" pattern
        3. Try last occurrence of single letter A-J
        """
        if not response:
            return None

        # Level 1: Try basic pattern
        pred = extract_answer_mmlu_pro(response)
        if pred is not None:
            return pred

        # Level 2: Try fallback patterns
        return extract_answer_again(response)

    def save_results(self, output_dir: str) -> None:
        """Save results to parquet file."""
        os.makedirs(output_dir, exist_ok=True)

        if not self.categories:
            cat_part = "all"
        elif len(self.categories) == 1:
            cat_part = self.categories[0]
        elif self.categories == self.STEM_CATEGORIES:
            cat_part = "stem"
        else:
            cat_part = f"{len(self.categories)}_categories"

        filename = f"{self.name}_{self.prompt_type}_{cat_part}.parquet"
        output_path = os.path.join(output_dir, filename)
        self.df.to_parquet(output_path, index=False)

        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import pdb; pdb.set_trace()

    benchmark = MMLUProBenchmark(
        prompt_type="cot",
        max_examples=10
    )
    
    print(benchmark.df['user_prompt'].iloc[0])
    print(benchmark.df['system_prompt'].iloc[0])
    print(benchmark.assistant_prompt)