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


class MMLUBenchmark(Benchmark):
    """MMLU (Massive Multitask Language Understanding) Benchmark.

    A comprehensive benchmark evaluating knowledge across 57 subjects
    spanning STEM, humanities, social sciences, and more.

    Paper: https://arxiv.org/abs/2009.03300
    Dataset: https://huggingface.co/datasets/cais/mmlu
    """

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    # All 57 subjects
    ALL_SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
        'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology', 'world_religions'
    ]

    STEM_SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security',
        'conceptual_physics', 'electrical_engineering', 'elementary_mathematics',
        'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
        'machine_learning', 'medical_genetics', 'virology'
    ]

    def __init__(
        self,
        subjects: Union[str, List[str]] = "all",
        prompt_type: str = "zero_shot",
        system_prompt: str = "You are a helpful assistant. Answer the following question to the best of your ability.",
        seed: int = 42,
        num_few_shot_examples: int = 5,
        max_examples: Optional[int] = None,
        name: str = "mmlu"
    ):
        """Initialize MMLU benchmark.

        Args:
            subjects: Which subject(s) to load. Can be:
                     - "all": all 57 subjects
                     - "stem": STEM subjects only
                     - List of subject names
            prompt_type: Type of prompting ('zero_shot' or 'few_shot')
            system_prompt: System prompt for the model
            seed: Random seed for reproducibility
            num_few_shot_examples: Number of examples for few-shot prompting (max 5 per subject)
            max_examples: Limit number of examples (for testing)
            name: Benchmark name
        """
        super().__init__(name)

        # Resolve subjects
        if subjects == "all":
            self.subjects = self.ALL_SUBJECTS
        elif subjects == "stem":
            self.subjects = self.STEM_SUBJECTS
        elif isinstance(subjects, list):
            self.subjects = subjects
            # Validate subjects
            for subject in self.subjects:
                if subject not in self.ALL_SUBJECTS:
                    raise ValueError(f"Invalid subject '{subject}'. Must be one of {self.ALL_SUBJECTS}")
        else:
            self.subjects = [subjects]
            if subjects not in self.ALL_SUBJECTS:
                raise ValueError(f"Invalid subject '{subjects}'. Must be one of {self.ALL_SUBJECTS}")

        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.num_few_shot_examples = min(num_few_shot_examples, 5)  # Max 5 per subject
        self.max_examples = max_examples

        valid_prompt_types = ['zero_shot', 'few_shot']
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: {valid_prompt_types}")

        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        """Load MMLU dataset from HuggingFace.

        For few-shot prompting, loads dev split (5 examples per subject).
        """
        random.seed(self.seed)

        # Load few-shot examples from dev split if needed
        if self.prompt_type == 'few_shot':
            self.dev_examples_by_subject = {}
            for subject in self.subjects:
                try:
                    dev_dataset = load_dataset('cais/mmlu', subject, split='dev')
                    self.dev_examples_by_subject[subject] = dev_dataset
                except Exception as e:
                    print(f"Warning: Could not load dev split for subject '{subject}': {e}")
                    self.dev_examples_by_subject[subject] = None

        # Load test data
        all_data = []
        for subject in self.subjects:
            try:
                test_dataset = load_dataset('cais/mmlu', subject, split='test')
                for example in test_dataset:
                    example['subject'] = subject
                    all_data.append(example)
            except Exception as e:
                print(f"Warning: Could not load test split for subject '{subject}': {e}")
                continue

        # Convert to dataframe
        rows = []
        for example in all_data:
            choices = example['choices']
            correct_idx = example['answer']
            correct_answer = choices[correct_idx]

            # Shuffle choices
            random.shuffle(choices)
            new_correct_idx = choices.index(correct_answer)
            correct_letter = self.INDEX_TO_LETTER[new_correct_idx]

            row_data = {
                'subject': example['subject'],
                'question': example['question'].strip(),
                'choice_A': choices[0].strip(),
                'choice_B': choices[1].strip(),
                'choice_C': choices[2].strip(),
                'choice_D': choices[3].strip(),
                'correct_answer': correct_letter,
                'correct_answer_text': correct_answer.strip(),
            }

            rows.append(row_data)

        self.df = pd.DataFrame(rows)

        # Apply max_examples limit if specified
        if self.max_examples:
            self.df = self.df.sample(n=min(self.max_examples, len(self.df)), random_state=self.seed)

        print(f"Loaded MMLU: {len(self.df)} test examples from {len(self.subjects)} subjects")

    def format_user_prompt(self) -> None:
        """Format prompts based on prompt_type."""
        prompts = []

        for _, row in self.df.iterrows():
            if self.prompt_type == 'zero_shot':
                prompt = self._zero_shot_prompt(row)
            elif self.prompt_type == 'few_shot':
                prompt = self._few_shot_prompt(row)
            else:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

            prompts.append(prompt)
        
        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def _base_prompt(self, row) -> str:
        """Base prompt format."""
        # subject_display = row['subject'].replace('_', ' ').title()
        # prompt = f"The following is a multiple choice question (with answers) about {subject_display}.\n\n"
        
        prompt = ""
        prompt += f"Question: {row['question']}\n"
        prompt += f"Choices:\n"
        prompt += f"(A) {row['choice_A']}\n"
        prompt += f"(B) {row['choice_B']}\n"
        prompt += f"(C) {row['choice_C']}\n"
        prompt += f"(D) {row['choice_D']}"
        return prompt

    def _zero_shot_prompt(self, row) -> str:
        """Zero-shot prompt - just the question without examples."""
        subject = row['subject']

        prompt = "Format your response as follows: \"The correct answer is (insert answer here)\", where the answer is one of A, B, C, or D.\n"
        prompt += f"The following is a multiple choice question (with answers) about {subject}.\n\n"
        prompt += self._base_prompt(row)

        self.assistant_prompt = "The correct answer is "
        return prompt

    def _few_shot_prompt(self, row) -> str:
        """Few-shot prompt - includes examples from dev split."""
        subject = row['subject']
        prompt = "Format your response as follows: \"The correct answer is (insert answer here)\", where the answer is one of A, B, C, or D.\n"
        prompt += f"The following are multiple choice questions (with answers) about {row['subject'].replace('_', ' ').title()}.\n\n"
        
        self.assistant_prompt = "The correct answer is "

        dev_examples = self.dev_examples_by_subject.get(subject)
        if dev_examples:
            n_examples = min(self.num_few_shot_examples, len(dev_examples))
            for i in range(n_examples):
                ex = dev_examples[i]
                choices = ex['choices']
                prompt += f"Question: {ex['question']}\n"
                prompt += f"Choices:\n"
                prompt += f"(A) {choices[0]}\n"
                prompt += f"(B) {choices[1]}\n"
                prompt += f"(C) {choices[2]}\n"
                prompt += f"(D) {choices[3]}\n"
                prompt += f"The correct answer is ({self.INDEX_TO_LETTER[ex['answer']]})\n\n"

        # Add test question
        prompt += f"Question: {row['question']}\n"
        prompt += f"Choices:\n"
        prompt += f"(A) {row['choice_A']}\n"
        prompt += f"(B) {row['choice_B']}\n"
        prompt += f"(C) {row['choice_C']}\n"
        prompt += f"(D) {row['choice_D']}\n"

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

        print(f"\nMMLU Results:")
        print(f"  Overall Accuracy: {accuracy:.2%} ({sum(scores)}/{len(scores)})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")

        # Print per-subject accuracy if multiple subjects
        if len(self.df['subject'].unique()) > 1:
            print(f"\n  Per-Subject Accuracy:")
            for subject in sorted(self.df['subject'].unique()):
                subject_df = self.df[self.df['subject'] == subject]
                subject_acc = subject_df['score'].mean()
                print(f"    {subject}: {subject_acc:.2%} ({subject_df['score'].sum()}/{len(subject_df)})")

    def _parse_answer(self, response: str) -> Optional[str]:
        """Extract answer choice (A/B/C/D) from model response."""
        if not response:
            return None

        # Try multiple patterns
        patterns = [
            r'answer is \((.)\)',
            r'Answer: \((.)\)',
            r'answer: \((.)\)',
            r'answer \((.)\)',
            r'\((.)\)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                if answer in self.LETTER_TO_INDEX:
                    return answer

        return None

    def save_results(self, output_dir: str) -> None:
        """Save results to parquet file."""
        os.makedirs(output_dir, exist_ok=True)

        if len(self.subjects) == 1:
            subjects_part = self.subjects[0]
        elif len(self.subjects) == len(self.ALL_SUBJECTS):
            subjects_part = "all"
        elif self.subjects == self.STEM_SUBJECTS:
            subjects_part = "stem"
        else:
            subjects_part = '_'.join(sorted(self.subjects)[:3])
            if len(self.subjects) > 3:
                subjects_part += f"_and_{len(self.subjects)-3}_more"

        filename = f"{self.name}_{subjects_part}_{self.prompt_type}.parquet"
        output_path = os.path.join(output_dir, filename)
        self.df.to_parquet(output_path, index=False)

        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import pdb; pdb.set_trace()

    benchmark = MMLUBenchmark(
        subjects="stem",
        prompt_type="few_shot",
        max_examples=10
    )
    print(benchmark.df['user_prompt'].iloc[0])
    print(benchmark.df['system_prompt'].iloc[0])
    print(benchmark.assistant_prompt)