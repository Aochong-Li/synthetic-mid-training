#!/usr/bin/env python3
"""
Split PMC open_access dataset into 'main' and 'auxiliary' splits.
- main: Articles with ==== Body tag, keep only body text
- auxiliary: Articles without proper structure tags

Uses filter() and map() for fast parallel processing.
Overwrites the cached dataset to save space (no duplicate copies).
"""

from datasets import load_dataset, DatasetDict
import argparse
import os
import shutil


def parse_body_text(text: str) -> tuple:
    """
    Parse text to extract body section.

    Returns:
        (has_body: bool, body_text: str)
    """
    if not text:
        return False, text

    # Check if text has the expected structure
    if '==== Body' not in text:
        return False, text

    try:
        # Split by Body marker
        parts = text.split('==== Body', 1)
        if len(parts) != 2:
            return False, text

        body_and_refs = parts[1]

        # Check if there's a Refs section and split it out
        if '==== Refs' in body_and_refs:
            body_text = body_and_refs.split('==== Refs', 1)[0]
        else:
            body_text = body_and_refs

        # Clean up the body text
        body_text = body_text.strip()

        if body_text:
            return True, body_text
        else:
            return False, text

    except Exception as e:
        print(f"Error parsing text: {e}")
        return False, text


def has_body_tag(example):
    """Check if example has ==== Body tag."""
    text = example.get('text', '')
    return '==== Body' in text


def extract_body_text(example):
    """Extract body text and overwrite text field."""
    text = example.get('text', '')
    has_body, body_text = parse_body_text(text)

    # Overwrite text with body
    example['text'] = body_text
    return example


def process_dataset(dataset_name: str = "pmc/open_access",
                    output_name: str = "pmc/open_access_split",
                    cache_dir: str = "/mnt/home/al2644/.cache/huggingface/datasets",
                    num_proc: int = 16):
    """
    Process PMC dataset and split into main and auxiliary.
    Saves to a new dataset location (does not overwrite original).

    Args:
        dataset_name: HuggingFace dataset name to load (e.g., "pmc/open_access")
        output_name: Name for the new split dataset (e.g., "pmc/open_access_split")
        cache_dir: HuggingFace cache directory
        num_proc: Number of processes for parallel processing
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    print(f"Original dataset size: {len(dataset['train']):,}")

    # Split dataset based on body tag presence
    print("\nFiltering dataset by body tag presence...")

    # Main split: has ==== Body tag
    print("Creating main split (with body)...")
    main_dataset = dataset['train'].filter(
        has_body_tag,
        num_proc=num_proc,
        desc="Filtering articles with body"
    )

    # Extract body text only
    print("Extracting body text...")
    main_dataset = main_dataset.map(
        extract_body_text,
        num_proc=num_proc,
        desc="Extracting body text"
    )

    # Auxiliary split: does NOT have ==== Body tag
    print("Creating auxiliary split (without body structure)...")
    auxiliary_dataset = dataset['train'].filter(
        lambda x: not has_body_tag(x),
        num_proc=num_proc,
        desc="Filtering articles without body"
    )

    print(f"\nSplit results:")
    print(f"  Main split (with body): {len(main_dataset):,}")
    print(f"  Auxiliary split (without structure): {len(auxiliary_dataset):,}")

    # Create new dataset with splits
    new_dataset = DatasetDict({
        'main': main_dataset,
        'auxiliary': auxiliary_dataset
    })

    # Determine cache path for new dataset
    # Convert output name to cache directory format (e.g., "pmc/open_access_split" -> "pmc___open_access_split")
    output_cache_name = output_name.replace('/', '___')
    output_path = os.path.join(cache_dir, output_cache_name)

    print(f"\nSaving new dataset to: {output_path}")

    # Save new dataset
    print("Saving dataset...")
    new_dataset.save_to_disk(output_path)

    print("\n" + "="*60)
    print("Done! New dataset saved.")
    print(f"Location: {output_path}")
    print("="*60)
    print(f"\nTo load the dataset:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_path}')")
    print(f"  main_split = dataset['main']  # Articles with body text")
    print(f"  auxiliary_split = dataset['auxiliary']  # Articles without structure")
    print(f"\nOriginal dataset '{dataset_name}' remains unchanged.")


def main():
    parser = argparse.ArgumentParser(
        description="Split PMC open_access dataset by body text availability"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pmc/open_access",
        help="HuggingFace dataset name to load (default: pmc/open_access)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="pmc/open_access_split",
        help="Name for output dataset (default: pmc/open_access_split)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/home/al2644/.cache/huggingface/datasets",
        help="HuggingFace cache directory (default: /mnt/home/al2644/.cache/huggingface/datasets)"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=16,
        help="Number of processes for parallel processing (default: 16)"
    )

    args = parser.parse_args()

    process_dataset(args.dataset, args.output_name, args.cache_dir, args.num_proc)


if __name__ == "__main__":
    main()
