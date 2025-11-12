#!/usr/bin/env python3
"""
Test downloader for PKG2 dataset - downloads only the first 5 files to verify setup.
"""

import os
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple
import logging

# Configuration
LINKS_FILE = "data/pubmed_kg/PKG2_links.txt"
OUTPUT_DIR = Path("/mnt/home/al2644/storage/pubmed_kg")
NUM_WORKERS = 3  # Use fewer workers for testing
TIMEOUT = 300  # 5 minutes per file
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
NUM_TEST_FILES = 5  # Only download first 5 files for testing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pkg2_download_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_filename_from_url(url: str) -> str:
    """Extract filename from the download URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    # Try to get fileName parameter
    if 'fileName' in params:
        return params['fileName'][0]

    # Fallback to path-based extraction
    if 'path' in params:
        path = params['path'][0]
        return os.path.basename(path)

    # Last resort: use fileId
    if 'fileId' in params:
        return f"file_{params['fileId'][0]}.gz"

    raise ValueError(f"Cannot extract filename from URL: {url}")


def download_file(url: str, output_path: Path, retries: int = MAX_RETRIES) -> Tuple[bool, str, str]:
    """
    Download a single file with retry logic.

    Returns:
        Tuple of (success, filename, error_message)
    """
    filename = output_path.name

    # Skip if file already exists
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 0:
            logger.info(f"⏭️  Skipping {filename} (already exists, {file_size:,} bytes)")
            return True, filename, "skipped"

    for attempt in range(retries):
        try:
            logger.info(f"📥 Downloading {filename} (attempt {attempt + 1}/{retries})")

            # Stream the download to handle large files
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))

            # Download to temporary file first
            temp_path = output_path.with_suffix(output_path.suffix + '.tmp')

            downloaded_size = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            # Rename temp file to final name
            temp_path.rename(output_path)

            size_mb = downloaded_size / (1024 * 1024)
            logger.info(f"✅ Downloaded {filename} ({size_mb:.2f} MB)")
            return True, filename, ""

        except requests.exceptions.Timeout:
            error_msg = f"Timeout downloading {filename}"
            logger.warning(f"⏱️  {error_msg}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return False, filename, error_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"Error downloading {filename}: {str(e)}"
            logger.warning(f"⚠️  {error_msg}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return False, filename, error_msg

        except Exception as e:
            error_msg = f"Unexpected error downloading {filename}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, filename, error_msg

    return False, filename, "Max retries exceeded"


def load_urls(links_file: str, limit: int = None) -> List[Tuple[str, str]]:
    """
    Load URLs from the links file.

    Returns:
        List of (url, filename) tuples
    """
    urls = []
    with open(links_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Remove line numbers if present (format: "123→https://...")
            if '→' in line:
                line = line.split('→', 1)[1]

            try:
                filename = parse_filename_from_url(line)
                urls.append((line, filename))

                # Limit for testing
                if limit and len(urls) >= limit:
                    break
            except ValueError as e:
                logger.warning(f"Skipping invalid URL: {e}")
                continue

    return urls


def main():
    """Main download orchestrator."""
    start_time = time.time()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"🧪 TEST MODE: Downloading only {NUM_TEST_FILES} files")

    # Load URLs
    logger.info(f"📋 Loading URLs from {LINKS_FILE}")
    url_list = load_urls(LINKS_FILE, limit=NUM_TEST_FILES)
    total_files = len(url_list)
    logger.info(f"📊 Found {total_files} files to download")

    if total_files == 0:
        logger.error("No URLs found in links file!")
        return 1

    # Download files in parallel
    logger.info(f"🚀 Starting parallel download with {NUM_WORKERS} workers")

    successful = 0
    failed = 0
    skipped = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_file, url, OUTPUT_DIR / filename): (url, filename)
            for url, filename in url_list
        }

        # Process completed downloads
        for future in as_completed(future_to_url):
            url, filename = future_to_url[future]
            try:
                success, fname, error = future.result()
                if success:
                    if error == "skipped":
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1
                    failed_files.append((fname, error))

                # Progress update
                completed = successful + failed + skipped
                progress = (completed / total_files) * 100
                logger.info(f"📈 Progress: {completed}/{total_files} ({progress:.1f}%) - ✅ {successful} | ⏭️ {skipped} | ❌ {failed}")

            except Exception as e:
                logger.error(f"Task exception for {filename}: {e}")
                failed += 1
                failed_files.append((filename, str(e)))

    # Summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("📊 TEST DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files:     {total_files}")
    logger.info(f"✅ Successful:   {successful}")
    logger.info(f"⏭️ Skipped:      {skipped}")
    logger.info(f"❌ Failed:       {failed}")
    logger.info(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
    logger.info("=" * 80)

    if failed_files:
        logger.error("\n❌ Failed downloads:")
        for fname, error in failed_files:
            logger.error(f"  - {fname}: {error}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob('*') if f.is_file())
    total_size_mb = total_size / (1024 ** 2)
    logger.info(f"\n💾 Total downloaded: {total_size_mb:.2f} MB")

    if failed == 0:
        logger.info("\n✅ Test successful! You can now run the full download script.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
