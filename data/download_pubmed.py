#!/usr/bin/env python3
"""
Download PubMed baseline XML files from NCBI FTP server with parallel workers.
Downloads files from pubmed25n0001.xml.gz to pubmed25n1274.xml.gz and unzips them.
"""

import os
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request
import urllib.error
from tqdm import tqdm
import argparse


def download_file(url, output_path, max_retries=3):
    """Download a file from URL to output_path with retry logic."""
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return True, f"Downloaded: {output_path.name}"
        except urllib.error.URLError as e:
            if attempt == max_retries - 1:
                return False, f"Failed to download {url}: {e}"
    return False, f"Failed to download {url} after {max_retries} attempts"


def unzip_file(gz_path, output_path):
    """Unzip a .gz file."""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True, f"Unzipped: {output_path.name}"
    except Exception as e:
        return False, f"Failed to unzip {gz_path}: {e}"


def download_and_unzip(file_number, base_url, output_dir, keep_gz=False):
    """Download and unzip a single PubMed file."""
    filename = f"pubmed25n{file_number:04d}.xml.gz"
    url = f"{base_url}/{filename}"
    gz_path = output_dir / filename
    xml_path = output_dir / filename.replace('.gz', '')

    # Skip if XML already exists
    if xml_path.exists():
        return True, f"Already exists: {xml_path.name}"

    # Download if .gz doesn't exist
    if not gz_path.exists():
        success, msg = download_file(url, gz_path)
        if not success:
            return False, msg

    # Unzip
    success, msg = unzip_file(gz_path, xml_path)

    # Remove .gz file if requested and unzip was successful
    if success and not keep_gz and gz_path.exists():
        gz_path.unlink()

    return success, msg


def main():
    """
    python data/download_pubmed.py --output-dir /mnt/home/al2644/storage/pubmed/baseline --workers 8 --start 1 --end 1274
    """
    parser = argparse.ArgumentParser(description='Download PubMed baseline XML files')
    parser.add_argument('--output-dir', type=str, default='./pubmed_baseline',
                        help='Output directory for downloaded files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel download workers')
    parser.add_argument('--start', type=int, default=1,
                        help='Start file number (default: 1)')
    parser.add_argument('--end', type=int, default=1274,
                        help='End file number (default: 1274)')
    parser.add_argument('--keep-gz', action='store_true',
                        help='Keep .gz files after unzipping')
    args = parser.parse_args()

    base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PubMed files {args.start} to {args.end}")
    print(f"Output directory: {output_dir}")
    print(f"Number of workers: {args.workers}")
    print(f"Keep .gz files: {args.keep_gz}")
    print("-" * 80)

    # Create list of file numbers to download
    file_numbers = list(range(args.start, args.end + 1))

    # Download and unzip files in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_num = {
            executor.submit(download_and_unzip, num, base_url, output_dir, args.keep_gz): num
            for num in file_numbers
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(file_numbers), desc="Processing files") as pbar:
            for future in as_completed(future_to_num):
                file_num = future_to_num[future]
                try:
                    success, msg = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        print(f"\n{msg}")
                except Exception as e:
                    failed += 1
                    print(f"\nError processing file {file_num}: {e}")
                pbar.update(1)

    print("-" * 80)
    print(f"Download complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(file_numbers)}")


if __name__ == "__main__":
    main()
