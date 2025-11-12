#!/usr/bin/env python3
"""
Test parallel processing with 128 samples and 32 workers.
"""

import sys
import warnings
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress fasttext warnings
warnings.filterwarnings('ignore')

from pubmed_simple_pipeline import SimplePubMedPipeline

def main():
    print("=" * 70)
    print("TESTING PARALLEL PROCESSING")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Samples: 128")
    print("  - Workers: 32")
    print("  - Output: /mnt/home/al2644/storage/pubmed/metadata")
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SimplePubMedPipeline(output_dir="/mnt/home/al2644/storage/pubmed/metadata")

    # Run with 32 workers on 128 samples
    print("\nStarting processing...")
    pipeline.process_dataset(max_records=128, num_workers=32)

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
