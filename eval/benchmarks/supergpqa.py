import os
import re
import sys
import random
import pandas as pd
from typing import Optional, List, Union
from datasets import load_dataset
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.benchmarks.benchmark import Benchmark

# Import SuperGPQA's evaluation functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tasks', 'SuperGPQA'))
from eval.eval import extract_option_labels, extract_option_content


class SuperGPQABenchmark(Benchmark):
    """SuperGPQA Benchmark.

    A comprehensive benchmark evaluating graduate-level knowledge and reasoning
    capabilities across 285 disciplines with 26,529 questions.

    Paper: https://arxiv.org/abs/2502.14739
    Dataset: https://huggingface.co/datasets/m-a-p/SuperGPQA
    Homepage: https://supergpqa.github.io/
    """

    # Support up to 10 options (A-J)
    LETTER_TO_INDEX = {chr(65 + i): i for i in range(10)}  # A-J
    INDEX_TO_LETTER = {i: chr(65 + i) for i in range(10)}

    # Five-shot examples (from config/prompt/five-shot.yaml)
    FIVE_SHOT_EXAMPLES = [
        {
            "question": "A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is",
            "options": ["10", "40", "6", "25", "15", "50", "30", "4", "5", "20"],
            "answer": "H",
            "reasoning": "In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4."
        },
        {
            "question": "Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?",
            "options": ["1000 times more", "50 times more", "5000 times more", "500 times more", "10000 times more", "20000 times more", "2000 times more", "100 times more", "10 times more", "N/A"],
            "answer": "E",
            "reasoning": "The amount of light a telescope can gather compared to the human eye is proportional to the area of its apertures. The area of a circle is given by the formula $A = \\pi \\left(\\frac{{D}}{{2}}\\right)^2$, where $D$ is the diameter. Therefore, the relative light-gathering power is calculated as:\n\\[\n\\frac{{\\left(\\frac{{50 \\text{{ cm}}}}{{2}}\\right)^2}}{{\\left(\\frac{{5 \\text{{ mm}}}}{{2}}\\right)^2}} = \\frac{{\\left(\\frac{{50 \\text{{ cm}}}}{{0.1 \\text{{ cm}}}}\\right)^2}}{{\\left(\\frac{{5 \\text{{ mm}}}}{{0.1 \\text{{ cm}}}}\\right)^2}} = \\frac{{500^2}}{{5^2}} = 10000.\n\\]"
        },
        {
            "question": "Where do most short-period comets come from and how do we know?",
            "options": [
                "The Kuiper belt; short period comets tend to be in the plane of the solar system like the Kuiper belt.",
                "The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.",
                "The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.",
                "The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.",
                "The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.",
                "The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.",
                "The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt."
            ],
            "answer": "A",
            "reasoning": "Most short-period comets originate from the Kuiper belt. This is deduced from the observation that these comets tend to follow orbits that lie in the plane of the solar system, similar to the distribution of objects in the Kuiper belt itself. Thus, the alignment of these cometary orbits with the ecliptic plane points to their Kuiper belt origin."
        },
        {
            "question": "Colors in a soap bubble result from light",
            "options": ["dispersion", "deflection", "refraction", "reflection", "interference", "converted to a different frequency", "polarization", "absorption", "diffraction", "transmission"],
            "answer": "E",
            "reasoning": "The colorful patterns observed in a soap bubble are caused by the phenomenon of light interference. This occurs when light waves bounce between the two surfaces of the soap film, combining constructively or destructively based on their phase differences and the varying thickness of the film. These interactions result in vibrant color patterns due to variations in the intensity of different wavelengths of light."
        },
        {
            "question": "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?",
            "options": ["240 W", "120 W", "10 W", "480 W", "360 W", "200 W", "30 W", "150 W", "60 W", "300 W"],
            "answer": "A",
            "reasoning": "The rate of energy usage, known as power, in an electrical circuit is calculated by the product of voltage and current. For a microwave oven connected to a 120 V outlet and drawing a current of 2 amps, the power consumption can be calculated as follows:\n\\[\n\\text{{Power}} = \\text{{Voltage}} \\times \\text{{Current}} = 120 \\, \\text{{V}} \\times 2 \\, \\text{{A}} = 240 \\, \\text{{W}}.\n\\]\nTherefore, the microwave oven uses energy at a rate of 240 watts."
        }
    ]
    
    MEDICAL_FIELDS = [
        'Traditional Chinese Medicine',
        'Clinical Medicine',
        'Basic Medicine',
        'Biology',
        'Public Health and Preventive Medicine',
        'Chemistry',
        'Chemical Engineering and Technology',
        'Pharmacy',
        'Environmental Science and Engineering',
        'Stomatology',
        'Aquaculture',
        'Food Science and Engineering',
        'Agricultural Engineering',
        'Animal Husbandry',
        'Crop Science',
        'Psychology',
        'Veterinary Medicine'
    ]
    
    def __init__(
        self,
        prompt_type: str = "zero_shot",
        system_prompt: str = "You are a helpful assistant. Answer the following question to the best of your ability.",
        seed: int = 42,
        max_examples: Optional[int] = None,
        fields: Union[List[str], str] = 'medical',
        name: str = "supergpqa"
    ):
        """Initialize SuperGPQA benchmark.

        Args:
            prompt_type: Type of prompting ('zero_shot' or 'five_shot')
            system_prompt: System prompt for the model
            seed: Random seed for reproducibility
            max_examples: Limit number of examples (for testing)
            fields: Filter by fields (list of field names)
            name: Benchmark name
        """
        super().__init__(name)

        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.max_examples = max_examples
        self.fields = fields
        
        # Validate prompt type
        valid_prompt_types = ['zero_shot', 'five_shot']
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: {valid_prompt_types}")

        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        """Load SuperGPQA dataset from HuggingFace."""
        random.seed(self.seed)
        
        dataset = load_dataset("m-a-p/SuperGPQA", split="train")
        df = dataset.to_pandas()

        if self.fields == 'medical':
            df = df[df['field'].isin(self.MEDICAL_FIELDS)]
        elif self.fields != 'all':
            df = df[df['field'].isin(self.fields)]

        if self.max_examples:
            df = df.sample(n=min(self.max_examples, len(df)), random_state=self.seed)

        rows = []
        for idx, row in df.iterrows():
            options = row['options']
            answer_letter = row['answer_letter']

            row_data = {
                'id': idx,
                'uuid': row['uuid'],
                'question': row['question'].strip(),
                'num_options': len(options),
                'correct_answer': answer_letter,
                'correct_answer_text': row['answer'].strip(),
                'discipline': row['discipline'],
                'field': row['field'],
                'subfield': row['subfield'],
                'difficulty': row['difficulty'],
                'is_calculation': row['is_calculation'],
            }

            for i, option in enumerate(options):
                row_data[f'choice_{chr(65+i)}'] = option.strip()
            rows.append(row_data)

        self.df = pd.DataFrame(rows)

        print(f"Loaded SuperGPQA: {len(self.df)} examples")
        if self.fields:
            print(f"  Filtered by fields: {self.fields}")

    def format_user_prompt(self) -> None:
        """Format prompts based on prompt_type."""
        prompts = []

        for _, row in self.df.iterrows():
            if self.prompt_type == 'zero_shot':
                prompt = self._zero_shot_prompt(row)
            elif self.prompt_type == 'five_shot':
                prompt = self._five_shot_prompt(row)
            else:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

            prompts.append(prompt)
        
        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def _format_question_with_options(self, question: str, options: List[str]) -> str:
        """Format question with options using A), B), C), etc."""
        formatted = question + "\n"
        for i, option in enumerate(options):
            formatted += f"{chr(65+i)}) {option}\n"
        return formatted

    def _zero_shot_prompt(self, row) -> str:
        """Zero-shot prompt - matches SuperGPQA zero-shot.yaml exactly."""
        # Get options for this question
        num_options = row['num_options']
        options = [row[f'choice_{chr(65+i)}'] for i in range(num_options)]

        prompt = "Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n"
        question_with_options = self._format_question_with_options(row['question'], options)
        prompt += question_with_options
        self.assistant_prompt = "Answer: "

        return prompt

    def _five_shot_prompt(self, row) -> str:
        """Five-shot prompt - matches SuperGPQA five-shot.yaml exactly."""
        prompt = "Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n"
        
        for ex in self.FIVE_SHOT_EXAMPLES:
            prompt += "Question: \n"
            prompt += self._format_question_with_options(ex['question'], ex['options'])
            prompt += f"\nThinking: Let's think step by step. {ex['reasoning']}"
            prompt += f"\nAnswer: {ex['answer']}.\n\n"

        num_options = row['num_options']
        options = [row[f'choice_{chr(65+i)}'] for i in range(num_options)]
        prompt += "Question: \n"
        prompt += self._format_question_with_options(row['question'], options)

        self.assistant_prompt = "Thinking: Let's think step by step. "
        return prompt

    def score(self) -> None:
        """Parse model responses and compute accuracy scores using SuperGPQA's exact logic."""
        scores = []
        parsed_answers = []

        for _, row in self.df.iterrows():
            response = row.get('response', '')
            correct_answer = row['correct_answer']
            num_options = row['num_options']

            # Get options list for fallback extraction
            options = [row[f'choice_{chr(65+i)}'] for i in range(num_options)]

            # Parse answer from response using SuperGPQA's exact extraction logic
            parsed = self._parse_answer(response, options)
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

        print(f"\nSuperGPQA Results:")
        print(f"  Overall Accuracy: {accuracy:.2%} ({sum(scores)}/{len(scores)})")
        print(f"  Refusal Rate: {refusal_rate:.2%}")

    def _parse_answer(self, response: str, options: List[str]) -> Optional[str]:
        """Extract answer choice using SuperGPQA's exact extraction logic from eval.py."""
        if not response:
            return None

        # Different logic for zero-shot vs five-shot (from eval.py lines 131-148)
        if self.prompt_type == 'zero_shot':
            # Try extract_option_labels first (always with 'ABCDEFGHIJ')
            predict = extract_option_labels(response, 'ABCDEFGHIJ')
            if predict is None or predict == 'error':
                # Fallback to extract_option_content
                predict = extract_option_content(response, options)
                if predict and predict in options:
                    # Convert content to letter
                    predict = chr(options.index(predict) + 65)
                else:
                    predict = None
            return predict

        elif self.prompt_type == 'five_shot':
            # First try on part before 'Question:' to avoid matching examples
            response_parts = response.split('Question:')
            if len(response_parts) > 0:
                response_first_part = response_parts[0]

                # Try extract_option_labels
                predict = extract_option_labels(response_first_part, 'ABCDEFGHIJ')
                if predict is None or predict == 'error':
                    # Try extract_option_content
                    predict = extract_option_content(response_first_part, options)
                    if predict and predict in options:
                        predict = chr(options.index(predict) + 65)
                    else:
                        predict = None

                # If still None, try full response
                if predict is None:
                    predict = extract_option_labels(response, 'ABCDEFGHIJ')
                    if predict is None or predict == 'error':
                        predict = extract_option_content(response, options)
                        if predict and predict in options:
                            predict = chr(options.index(predict) + 65)
                        else:
                            predict = None

                return predict
            else:
                # Shouldn't happen, but fallback to zero-shot logic
                predict = extract_option_labels(response, 'ABCDEFGHIJ')
                if predict is None or predict == 'error':
                    predict = extract_option_content(response, options)
                    if predict and predict in options:
                        predict = chr(options.index(predict) + 65)
                    else:
                        predict = None
                return predict

        return None

    def save_results(self, output_dir: str) -> None:
        """Save results to parquet file."""
        os.makedirs(output_dir, exist_ok=True)
        if self.fields:
            if isinstance(self.fields, list):
                fields_part = '_'.join(sorted(self.fields))
            else:
                fields_part = self.fields
        else:
            fields_part = "all"

        filename = f"{self.name}_{self.prompt_type}_fields_{fields_part}"
        filename += ".parquet"

        output_path = os.path.join(output_dir, filename)
        self.df.to_parquet(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import pdb; pdb.set_trace()

    benchmark = SuperGPQABenchmark(
        prompt_type="five_shot",
        max_examples=10
    )
    
    print(benchmark.df['user_prompt'][0])
    print(benchmark.df['system_prompt'][0])
    print(benchmark.assistant_prompt)
