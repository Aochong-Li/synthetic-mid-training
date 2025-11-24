import os
import re
import ast
import random
import pandas as pd
from typing import Optional, List, Union
from datasets import load_dataset
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.benchmarks.benchmark import Benchmark

class MMLUReduxBenchmark(Benchmark):
    """MMLU-Redux Benchmark.

    A carefully annotated version of MMLU (Massive Multitask Language Understanding)
    to provide a more accurate and reliable benchmark. MMLU-Redux consists of 30 MMLU
    subjects, each containing 100 randomly sampled questions.

    Paper: https://arxiv.org/pdf/2406.04127
    Dataset: https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
    """

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    MEDICAL_SUBJECTS = [
        "anatomy",
        "clinical_knowledge",
        "college_biology",
        "high_school_biology",
        "virology",
        "medical_genetics",
        "college_medicine",
        "professional_medicine",
        "human_aging",
        "nutrition",
        "human_sexuality",
        "professional_psychology",
        "high_school_psychology",
        "college_chemistry",
        "high_school_chemistry",
        "college_physics",
        "conceptual_physics",
        "high_school_physics"
    ]
    STEM_SUBJECTS = [
        'anatomy',
        'clinical_knowledge',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'electrical_engineering',
        'high_school_chemistry',
        'high_school_mathematics',
        'high_school_physics',
        'high_school_statistics',
        'machine_learning',
        'virology',
        'conceptual_physics',
        'astronomy',
        'econometrics'
    ]

    def __init__(
        self,
        subjects: Union[str, List[str]] = "all",
        prompt_type: str = "zero_shot",
        system_prompt: str = "You are a helpful assistant. Answer the following question to the best of your ability.",
        seed: int = 42,
        max_examples: Optional[int] = None,
        data_dir: Optional[str] = None,
        name: str = "mmlu_redux"
    ):
        """Initialize MMLU-Redux benchmark.

        Args:
            subjects: Which subject(s) to load. Can be a string or list of strings.
                     Use "all" to load all available subjects.
            prompt_type: Type of prompting ('zero_shot' or 'few_shot')
            system_prompt: System prompt for the model
            seed: Random seed for reproducibility
            max_examples: Limit number of examples per subject (for testing)
            data_dir: Directory containing the MMLU-Redux CSV files.
                     If None, uses eval/tasks/mmlu-redux/mmlu_redux/
            name: Benchmark name
        """
        super().__init__(name)

        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(__file__), '..', 'tasks', 'mmlu-redux', 'mmlu_redux'
            )
        else:
            self.data_dir = data_dir

        self.available_subjects = self._get_available_subjects()

        if subjects == "all" or subjects == ["all"]:
            self.subjects = self.available_subjects
        elif subjects == "medical" or subjects == ["medical"]:
            self.subjects = self.MEDICAL_SUBJECTS
        elif subjects == "stem" or subjects == ["stem"]:
            self.subjects = self.STEM_SUBJECTS
        else:
            self.subjects = [subjects]
        
        for subject in self.subjects:
            if subject not in self.available_subjects:
                raise ValueError(
                    f"Invalid subject '{subject}'. Available subjects: {self.available_subjects}"
                )

        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.max_examples = max_examples
        
        valid_prompt_types = ['zero_shot', 'few_shot']
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: {valid_prompt_types}")

        self.load_dataset()
        self.format_user_prompt()

    def _get_available_subjects(self) -> List[str]:
        """Get list of available subjects from the data directory."""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        subjects = []
        for f in csv_files:
            if f.startswith('mmlu_'):
                subject = f[5:-4]
                subjects.append(subject)
        return sorted(subjects)
    
    def remove_error_rows(self, row: pd.Series):
        if str(row['error_type']).strip().upper() == 'OK':
            return row
        else:
            row['answer'] = None
            return row

    def load_dataset(self) -> None:
        """Load MMLU-Redux dataset from CSV files.

        For few-shot prompting, loads dev split from cais/mmlu (5 examples per subject).

        The CSV format includes:
        - question: The question text
        - choices: String representation of list with 4 choices
        - answer: Index of correct answer (0-3)
        - error_type: "OK" or error type description
        - source: URL source
        - correct_answer: (optional) correct answer index if error_type != "OK"
        - potential_reason: (optional) reason for error
        """
        random.seed(self.seed)

        all_data = []

        # Load few-shot examples from cais/mmlu dev split if needed
        if self.prompt_type == 'few_shot':
            self.few_shot_examples_dict = {}  # Store by subject
            for subject in self.subjects:
                try:
                    # Load dev split from cais/mmlu (contains 5 examples)
                    dev_dataset = load_dataset('cais/mmlu', subject, split='dev')
                    self.few_shot_examples_dict[subject] = dev_dataset
                except Exception as e:
                    print(f"Warning: Could not load dev split for subject '{subject}': {e}")
                    self.few_shot_examples_dict[subject] = None
        else:
            self.few_shot_examples_dict = {}

        # Load test data from MMLU-Redux CSV files
        for subject in self.subjects:
            csv_path = os.path.join(self.data_dir, f"mmlu_{subject}.csv")

            if not os.path.exists(csv_path):
                print(f"Warning: File not found for subject '{subject}': {csv_path}")
                continue

            subject_df = pd.read_csv(csv_path, sep=None, engine='python')
            subject_df['choices'] = subject_df['choices'].apply(ast.literal_eval)
            subject_df['subject'] = subject

            all_data.append(subject_df)

        combined_df = pd.concat(all_data, ignore_index=True).apply(self.remove_error_rows, axis=1).dropna(subset=['answer'])

        if self.max_examples:
            combined_df = combined_df.sample(self.max_examples, random_state=self.seed)

        rows = []
        for idx, row in combined_df.iterrows():
            choices = row['choices']
            correct_idx = int(row['answer'])
            correct_answer = choices[correct_idx]
            
            random.shuffle(choices)
            correct_idx = choices.index(correct_answer)
            correct_letter = self.INDEX_TO_LETTER[correct_idx]

            row_data = {
                'id': idx,
                'subject': row['subject'],
                'question': row['question'].strip(),
                'choice_A': choices[0].strip(),
                'choice_B': choices[1].strip(),
                'choice_C': choices[2].strip(),
                'choice_D': choices[3].strip(),
                'correct_answer': correct_letter,
                'correct_answer_text': correct_answer.strip(),
            }

            rows.append(row_data)

        self.df = pd.DataFrame(rows)
        print(f"Loaded MMLU-Redux: {len(self.df)} examples from {len(self.subjects)} subjects")
        
    def format_user_prompt(self) -> None:
        """Format prompts based on prompt_type."""
        prompts = []

        for _, row in self.df.iterrows():
            # Format prompt based on type
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
        """Base prompt format for MMLU-Redux."""
        prompt = ""
        prompt += f"Question: {row['question']}\n"
        prompt += f"Choices:\n(A) {row['choice_A']}\n(B) {row['choice_B']}\n(C) {row['choice_C']}\n(D) {row['choice_D']}"
        return prompt

    def _zero_shot_prompt(self, row) -> str:
        """Zero-shot prompt - just the question without examples."""
        subject_display = row['subject'].replace('_', ' ').title()

        prompt = "Format your response as follows: \"The correct answer is (insert answer here)\", where the answer is one of A, B, C, or D.\n"
        prompt += f"The following is a multiple choice question (with answers) about {subject_display}.\n\n"
        prompt += self._base_prompt(row)

        self.assistant_prompt = "The correct answer is "
        return prompt

    def _few_shot_prompt(self, row) -> str:
        """Few-shot prompt - includes 5 examples from cais/mmlu dev split."""
        subject = row['subject']
        subject_display = row['subject'].replace('_', ' ').title() 

        prompt = "Format your response as follows: \"The correct answer is (insert answer here)\".\n"
        prompt += f"The following are multiple choice questions (with answers) about {subject_display}.\n\n"

        dev_examples = self.few_shot_examples_dict.get(subject)

        if dev_examples:
            for ex in dev_examples:
                choices = ex['choices']
                prompt += f"Question: {ex['question']}\n"
                prompt += f"Choices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n"
                prompt += f'The correct answer is ({self.INDEX_TO_LETTER[ex["answer"]]})\n\n'

        prompt += self._base_prompt(row)
        
        self.assistant_prompt = "The correct answer is "
        return prompt

    def score(self) -> None:
        """Parse model responses and compute accuracy scores.

        Parses the model's response to extract the answer choice (A/B/C/D)
        and compares it to the correct answer.
        """
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

        print(f"\nMMLU-Redux Results:")
        print(f"  Overall Accuracy: {accuracy:.2%} ({sum(scores)}/{len(scores)})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")

    def _parse_answer(self, response: str) -> Optional[str]:
        """Extract answer choice (A/B/C/D) from model response.

        Looks for various patterns where the model indicates its answer.
        Returns the first valid letter found, or None if no valid answer.
        """
        if not response:
            return None

        # Clean up the response
        response = response.strip()
        patterns = [
            r'answer is \(([A-J])\)',
            r'Answer: \(([A-J])\)',
            r'answer: \(([A-J])\)',
            r'answer \(([A-J])\)',
            r'\(([A-J])\)'
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
        elif len(self.subjects) == len(self.available_subjects):
            subjects_part = "all"
        elif self.subjects == self.MEDICAL_SUBJECTS:
            subjects_part = "medical"
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

    benchmark = MMLUReduxBenchmark(
        subjects="medical",
        prompt_type="few_shot",
        max_examples=10
    )
    print(benchmark.df['user_prompt'].iloc[0])
    print(benchmark.df['system_prompt'].iloc[0])
    print(benchmark.assistant_prompt)