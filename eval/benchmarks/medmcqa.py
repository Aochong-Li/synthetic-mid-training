import os
import re
import json
import random
import pandas as pd
from typing import Optional, Union, List
from collections import namedtuple
from datasets import load_dataset
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.benchmarks.benchmark import Benchmark

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index', 'explanation', 'subject_name'])

class MedMCQABenchmark(Benchmark):
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    SUBJECTS = [
        'Anatomy', 'Medicine', 'Pathology', 'Surgery', 'Pharmacology', 'Social & Preventive Medicine',
        'Microbiology', 'Physiology', 'Gynaecology & Obstetrics', 'Biochemistry', 'Pediatrics', 'Ophthalmology',
        'ENT', 'Forensic Medicine', 'Orthopaedics', 'Psychiatry', 'Radiology', 'Anaesthesia', 'Dental', 'Skin'
        ]
    FEW_SHOT_EXAMPLES =[
                        {
                        "question": "Most commonly used route of administration of heparin for post-operative thromboprophylaxis is?",
                        "choice_A": "Subcutaneous",
                        "choice_B": "Intravenous",
                        "choice_C": "Inhalational",
                        "choice_D": "Intramuscular",
                        "correct_answer": "A",
                        "correct_answer_text": "Subcutaneous",
                        "explanation": "UFH is injected s.c. every 8–12 hours, started before surgery and continued for 7–10 days, or till the patient starts moving about. This regimen has been found to prevent postoperative deep vein thrombosis (postoperative thromboprophylaxis) without increasing surgical bleeding. It also does not prolong aPTT or clotting time. However, it should not be used in case of neurosurgery or when spinal anaesthesia is to be given.",
                        "subject_name": "Pharmacology"
                        },
                        {
                        "question": "A young female presents to OPD with a spontaneous abortion and secondary amenorrhea since then. FSH was found to be 6 IU/mL. What is the most probable cause of amenorrhea?",
                        "choice_A": "Ovarian failure",
                        "choice_B": "Pituitary failure",
                        "choice_C": "Ongoing pregnancy",
                        "choice_D": "Uterine synechiae",
                        "correct_answer": "D",
                        "correct_answer_text": "Uterine synechiae",
                        "explanation": "Since the lady is having secondary amenorrhea following an abortion, uterine synechiae is the most likely. Although uterine synechiae mostly develops with an overzealous curettage, it is also seen in spontaneous abortions. Also, here FSH levels are normal so it points to an end-organ pathology. (Normal serum FSH value in adult woman is 5–20 IU/mL.) In case of ovarian failure—FSH will be high. In case of pituitary failure—level of FSH is low.",
                        "subject_name": "Gynaecology & Obstetrics"
                        },
                        {
                        "question": "Which of the following is not used in heart failure?",
                        "choice_A": "Metoprolol",
                        "choice_B": "Trimetazidine",
                        "choice_C": "Sacubitril",
                        "choice_D": "Nesiritide",
                        "correct_answer": "B",
                        "correct_answer_text": "Trimetazidine",
                        "explanation": "Beta blockers in heart failure—Beta blockers are contraindicated in acute heart failure but they can be used in chronic heart failure. At first, beta blockers should be started at low dose; dose should be increased gradually. Beta blockers used are carvedilol, metoprolol and bisoprolol. Sacubitril—It is a NEP (neprilysin) inhibitor, which is required for metabolism of BNP (brain natriuretic peptide); as a result BNP levels are increased, resulting in natriuresis and vasodilation; thus can be used in CHF. Nesiritide—It is recombinant BNP. It is given through intravenous route. Trimetazidine—It is a metabolic modulator. It partially inhibits β-oxidation of fatty acids, which results in shifting of metabolism of heart muscles from fatty acids to glucose, which requires less oxygen, so beneficial for patients with angina pectoris but not used in heart failure.",
                        "subject_name": "Pharmacology"
                        },
                        {
                        "question": "Which of the following muscle do NOT work for inversion of foot?",
                        "choice_A": "Extensor hallucis longus",
                        "choice_B": "Tibialis anterior",
                        "choice_C": "Tibialis posterior",
                        "choice_D": "Peroneus longus",
                        "correct_answer": "D",
                        "correct_answer_text": "Peroneus longus",
                        "explanation": "Movement — Muscles — Accessory muscles. INVERSION: Tibialis anterior, Tibialis posterior; accessory: Extensor hallucis longus, Flexor digitorum longus, Flexor hallucis longus. EVERSION: Peroneus longus, Peroneus brevis, Peroneus tertius.",
                        "subject_name": "Anatomy"
                        },
                        {
                        "question": "Slow growing alveolar like tumor in liver",
                        "choice_A": "E. granulosus",
                        "choice_B": "E. multilocularis",
                        "choice_C": "Cysticercus cellulosae",
                        "choice_D": "Amoebic liver abscess",
                        "correct_answer": "B",
                        "correct_answer_text": "E. multilocularis",
                        "explanation": "Echinococcus multilocularis causes malignant hydatid disease in which tumors are slow-growing and are alveoli-like—ill-defined, slow-growing and invasive. It causes alveolar hydatidosis (alveolar echinococcosis). Echinococcus granulosus causes hydatid disease in liver or cystic echinococcosis. Amoebic liver abscess is an extraintestinal manifestation of Entamoeba histolytica. Cysticercus cellulosae is the larval stage of Taenia solium.",
                        "subject_name": "Microbiology"
                        }
                    ]

    def __init__(
        self,
        prompt_type: str = "chain_of_thought",
        system_prompt: str = "You are a helpful assistant with expertise in medicine. Answer the following question to the best of your ability.",
        seed: int = 42,
        num_few_shot_examples: int = 5,
        samples_per_subject: int = 600,
        name: str = "medmcqa"
    ):
        super().__init__(name)

        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.seed = seed
        self.num_few_shot_examples = num_few_shot_examples
        self.samples_per_subject = samples_per_subject
        
        self.load_dataset()
        self.format_user_prompt()

    def load_dataset(self) -> None:
        random.seed(self.seed)
        '''
        This is an example of MedMCQA dataset:
            {
                "question":"A 40-year-old man presents with 5 days of productive cough and fever. Pseudomonas aeruginosa is isolated from a pulmonary abscess. CBC shows an acute effect characterized by marked leukocytosis (50,000 mL) and the differential count reveals a shift to left in granulocytes. Which of the following terms best describes these hematologic findings?",
                "exp": "Circulating levels of leukocytes and their precursors may occasionally reach very high levels (>50,000 WBC mL). These extreme elevations are sometimes called leukemoid reactions because they are similar to the white cell counts observed in leukemia, from which they must be distinguished. The leukocytosis occurs initially because of the accelerated release of granulocytes from the bone marrow (caused by cytokines, including TNF and IL-1) There is a rise in the number of both mature and immature neutrophils in the blood, referred to as a shift to the left. In contrast to bacterial infections, viral infections (including infectious mononucleosis) are characterized by lymphocytosis Parasitic infestations and certain allergic reactions cause eosinophilia, an increase in the number of circulating eosinophils. Leukopenia is defined as an absolute decrease in the circulating WBC count.",
                "cop":1,
                "opa":"Leukemoid reaction",
                "opb":"Leukopenia",
                "opc":"Myeloid metaplasia",
                "opd":"Neutrophilia",
                "subject_name":"Pathology",
                "topic_name":"Basic Concepts and Vascular changes of Acute Inflammation",
                "id":"4e1715fe-0bc3-494e-b6eb-2d4617245aef",
                "choice_type":"single"
            }
        '''

        medmcqa_dataset = load_dataset("openlifescienceai/medmcqa")
        key_columns = ['question', 'opa', 'opb', 'opc', 'opd', 'cop', 'subject_name', 'topic_name']

        # Use train split as test data (test split doesn't have valid cop labels)
        train_df = medmcqa_dataset['train'].to_pandas()

        # Filter: drop rows with None topic_name and only keep valid subjects
        train_df = train_df.dropna(subset=key_columns)
        train_df = train_df[train_df['subject_name'].isin(self.SUBJECTS)]

        # Sample N examples per subject
        sampled_dfs = []
        for subject in self.SUBJECTS:
            subject_df = train_df[train_df['subject_name'] == subject]
            if len(subject_df) > 0:
                n_samples = min(self.samples_per_subject, len(subject_df))
                sampled = subject_df.sample(n=n_samples, random_state=self.seed)
                sampled_dfs.append(sampled)

        test_df = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()

        rows = []
        for idx, row in test_df.iterrows():
            choices = [
                row['opa'],
                row['opb'],
                row['opc'],
                row['opd']
            ]
            correct_answer = choices[row['cop']]
            random.shuffle(choices)

            correct_idx = choices.index(correct_answer)
            correct_letter = self.INDEX_TO_LETTER[correct_idx]

            row_data = {
                'id': idx,
                'question': row['question'].strip(),
                'choice_A': choices[0].strip(),
                'choice_B': choices[1].strip(),
                'choice_C': choices[2].strip(),
                'choice_D': choices[3].strip(),
                'correct_answer': correct_letter,
                'correct_answer_text': correct_answer.strip(),
                'subject_name': row['subject_name'].strip(),
                'topic_name': row['topic_name'].strip(),
                'question_id': row['id'],
                'choice_type': row['choice_type'],
                'explanation': row.get('exp', ''),
            }

            rows.append(row_data)

        self.df = pd.DataFrame(rows)

        print(f"Loaded MedMCQA: {len(self.df)} test examples from train split")
        print(f"  Sampled {self.samples_per_subject} examples per subject (where available)")

    def format_user_prompt(self) -> None:
        prompts = []

        for _, row in self.df.iterrows():
            example = Example(
                question=row['question'],
                choice1=row['choice_A'],
                choice2=row['choice_B'],
                choice3=row['choice_C'],
                choice4=row['choice_D'],
                correct_index=self.LETTER_TO_INDEX[row['correct_answer']],
                explanation=row['explanation'],
                subject_name=row['subject_name'],
            )

            if self.prompt_type == 'zero_shot':
                prompt = self._zero_shot_prompt(example)
            elif self.prompt_type == 'chain_of_thought':
                prompt = self._chain_of_thought_prompt(example)
            elif self.prompt_type == '5_shot':
                prompt = self._five_shot_prompt(example)
            else:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

            prompts.append(prompt)
            
        self.df['system_prompt'] = self.system_prompt
        self.df['user_prompt'] = prompts

    def _base_prompt(self, example: Example) -> str:
        prompt = f"Question: {example.question}\n"
        prompt += f"Choices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        return prompt

    def _zero_shot_prompt(self, example: Example) -> str:
        prompt = f"Format your response as follows: \"The correct answer is (insert answer here)\"\n\n"
        prompt += self._base_prompt(example)
        self.assistant_prompt = "The correct answer is "
        
        return prompt

    def _chain_of_thought_prompt(self, example: Example) -> str:
        prompt = "Here are some example questions and answers with explanations. Answer the final question yourself, giving your reasoning beforehand. Format your response as follows: \"The correct answer is (insert answer here)\"\n\n"
        
        for ex in self.FEW_SHOT_EXAMPLES[:self.num_few_shot_examples]:
            prompt += f'Question: {ex["question"]}\n'
            prompt += f'Choices:\n'
            prompt += f'(A) {ex["choice_A"]}\n'
            prompt += f'(B) {ex["choice_B"]}\n'
            prompt += f'(C) {ex["choice_C"]}\n'
            prompt += f'(D) {ex["choice_D"]}\n'
            prompt += f"Let's think step by step: {ex['explanation']}\n"
            prompt += f'The correct answer is ({ex["correct_answer"]})\n\n'

        prompt += f"Question: {example.question}\n"
        prompt += f"Choices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"

        self.assistant_prompt = "Let's think step by step: "
        
        return prompt

    def _five_shot_prompt(self, example: Example) -> str:
        prompt = "Here are some example questions and answers. Answer the final question yourself. Format your response as follows: \"The correct answer is (insert answer here)\"\n\n"
        for ex in self.FEW_SHOT_EXAMPLES[:self.num_few_shot_examples]:
            prompt += f'Question: {ex["question"]}\n'
            prompt += f'Choices:\n'
            prompt += f'(A) {ex["choice_A"]}\n'
            prompt += f'(B) {ex["choice_B"]}\n'
            prompt += f'(C) {ex["choice_C"]}\n'
            prompt += f'(D) {ex["choice_D"]}\n'
            prompt += f'The correct answer is ({ex["correct_answer"]})\n\n'

        prompt += f"Question: {example.question}\n"
        prompt += f"Choices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"

        self.assistant_prompt = "The correct answer is "
        
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
        print(f"\nMedMCQA Results:")
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

    benchmark = MedMCQABenchmark(
        prompt_type="chain_of_thought",
        samples_per_subject=10,
        seed=42
    )
    
    print(benchmark.df['user_prompt'].iloc[0])
    print(benchmark.df['system_prompt'].iloc[0])
    print(benchmark.assistant_prompt)