#!/usr/bin/env python3
"""
Simplified PubMed metadata pipeline.
Fetches metadata from PubMed and stores in SQLite database.
"""

import sqlite3
import json
import time
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# Suppress FastText warnings globally
warnings.filterwarnings('ignore')

from filter_pubmed import PubMedProcessor


# Global processor for each worker (initialized once per worker process)
_worker_processor = None


def _init_worker(fasttext_path):
    """Initialize worker with a processor instance (called once per worker)."""
    global _worker_processor
    import warnings
    import os
    warnings.filterwarnings('ignore')
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        _worker_processor = PubMedProcessor(fasttext_model_path=fasttext_path)
    except Exception as e:
        print(f"[Worker {os.getpid()}] ERROR loading FastText: {e}", flush=True)
        raise


def _worker_process_pmid(item, db_path):
    """Worker function to process a single PMID (must be at module level for pickle)."""
    global _worker_processor
    pmid, pmc_index = item["pmid"], item["pmc_index"]

    # Check if processor was initialized
    if _worker_processor is None:
        error_msg = "Worker processor not initialized!"
        print(f"ERROR: {error_msg} for PMID {pmid}", flush=True)
        with sqlite3.connect(db_path, timeout=300.0) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pubmedmetadata (pmid, pmc_index, fetched_at, status, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (pmid, pmc_index, datetime.now().isoformat(), "error", error_msg))
            conn.commit()
        return False

    processor = _worker_processor

    try:
        # Fetch metadata (no rate limit)
        records = processor.fetch_metadata(pmid)

        if not records:
            with sqlite3.connect(db_path, timeout=300.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pubmedmetadata (pmid, pmc_index, fetched_at, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (pmid, pmc_index, datetime.now().isoformat(), "error", "No records returned"))
                conn.commit()
            return False

        # Insert articles
        with sqlite3.connect(db_path, timeout=300.0) as conn:
            for record in records:
                conn.execute("""
                    INSERT OR REPLACE INTO pubmedmetadata (
                        pmid, article_title, abstract, language,
                        journal_title, journal_issn,
                        pub_year, pub_month, pub_day,
                        article_ids, authors, mesh_headings, keywords, article_references,
                        citation_count, pmc_index, fetched_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.get("pmid"),
                    record.get("article_title"),
                    record.get("abstract"),
                    record.get("language"),
                    record.get("journal_title"),
                    record.get("journal_issn"),
                    record.get("pub_date_year"),
                    record.get("pub_date_month"),
                    record.get("pub_date_day"),
                    json.dumps(record.get("article_ids", {})),
                    json.dumps(record.get("authors", [])),
                    json.dumps(record.get("mesh_headings", [])),
                    json.dumps(record.get("keywords", [])),
                    json.dumps(record.get("references", [])),
                    len(record.get("references", [])),
                    pmc_index,
                    datetime.now().isoformat(),
                    "success"
                ))
            conn.commit()
        return True

    except Exception as e:
        # Insert error
        try:
            with sqlite3.connect(db_path, timeout=300.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pubmedmetadata (pmid, pmc_index, fetched_at, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (pmid, pmc_index, datetime.now().isoformat(), "error", str(e)))
                conn.commit()
        except:
            pass
        return False


class SimplePubMedPipeline:
    """Simple pipeline for fetching and storing PubMed metadata."""

    def __init__(self, output_dir: str = "/mnt/home/al2644/storage/pubmed/metadata"):
        """
        Initialize pipeline.

        Args:
            output_dir: Directory to store database and logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.output_dir / "pubmed.db"
        self.processor = PubMedProcessor(
            fasttext_model_path="/mnt/home/al2644/storage/fasttext/models/lid.176.bin"
        )

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize database
        self._init_db()

        # Statistics
        self.stats = {"processed": 0, "success": 0, "errors": 0, "skipped": 0}

    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            # Aggressive WAL mode settings for high concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=100000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=30000000000")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS pubmedmetadata (
                    pmid INTEGER PRIMARY KEY,

                    -- Article info
                    article_title TEXT,
                    abstract TEXT,
                    language TEXT,

                    -- Journal info
                    journal_title TEXT,
                    journal_issn TEXT,

                    -- Dates
                    pub_year TEXT,
                    pub_month TEXT,
                    pub_day TEXT,

                    -- JSON fields
                    article_ids TEXT,
                    authors TEXT,
                    mesh_headings TEXT,
                    keywords TEXT,
                    article_references TEXT,

                    -- Metadata
                    citation_count INTEGER DEFAULT 0,
                    pmc_index INTEGER,
                    fetched_at TIMESTAMP,
                    status TEXT,
                    error_message TEXT
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON pubmedmetadata(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON pubmedmetadata(pub_year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal ON pubmedmetadata(journal_title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_citations ON pubmedmetadata(citation_count)")

            conn.commit()

        self.logger.info(f"Database initialized at {self.db_path} (WAL mode, 5min timeout, aggressive cache)")

    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if error should be retried."""
        error_lower = error_msg.lower()
        retryable = ["ssl", "connection", "timeout", "max retries", "429", "500", "502", "503"]
        return any(keyword in error_lower for keyword in retryable)

    def process_pmid(self, pmid: int, pmc_index: int = None, retry_count: int = 0) -> bool:
        """
        Fetch and store metadata for a single PMID.

        Args:
            pmid: PubMed ID
            pmc_index: Index in PMC dataset
            retry_count: Current retry attempt

        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch metadata
            records = self.processor.fetch_metadata(pmid)

            if not records:
                self._insert_error(pmid, "No records returned", pmc_index)
                return False

            # Insert into database
            for record in records:
                self._insert_article(record, pmc_index)

            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"PMID {pmid}: {error_msg}")

            # Retry logic for network errors
            if self._is_retryable_error(error_msg) and retry_count < 5:
                wait_time = [1, 5, 30, 30, 30][retry_count]
                self.logger.info(f"Retrying PMID {pmid} in {wait_time}s (attempt {retry_count + 1}/5)")
                time.sleep(wait_time)
                return self.process_pmid(pmid, pmc_index, retry_count + 1)

            # Give up and log error
            self._insert_error(pmid, error_msg, pmc_index)
            return False

    def _insert_article(self, record: Dict[str, Any], pmc_index: int = None):
        """Insert article record into database."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pubmedmetadata (
                    pmid, article_title, abstract, language,
                    journal_title, journal_issn,
                    pub_year, pub_month, pub_day,
                    article_ids, authors, mesh_headings, keywords, article_references,
                    citation_count, pmc_index, fetched_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("pmid"),
                record.get("article_title"),
                record.get("abstract"),
                record.get("language"),
                record.get("journal_title"),
                record.get("journal_issn"),
                record.get("pub_date_year"),
                record.get("pub_date_month"),
                record.get("pub_date_day"),
                json.dumps(record.get("article_ids", {})),
                json.dumps(record.get("authors", [])),
                json.dumps(record.get("mesh_headings", [])),
                json.dumps(record.get("keywords", [])),
                json.dumps(record.get("references", [])),
                len(record.get("references", [])),
                pmc_index,
                datetime.now().isoformat(),
                "success"
            ))
            conn.commit()

    def _insert_error(self, pmid: int, error_msg: str, pmc_index: int = None):
        """Insert error record into database."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pubmedmetadata (pmid, pmc_index, fetched_at, status, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (pmid, pmc_index, datetime.now().isoformat(), "error", error_msg))
            conn.commit()

    def get_processed_pmids(self) -> set:
        """Get set of already processed PMIDs."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            cursor = conn.execute("SELECT pmid FROM pubmedmetadata")
            return {row[0] for row in cursor.fetchall()}

    def process_dataset(self, dataset_name: str = "pmc/open_access", max_records: int = None, num_workers: int = 1):
        """
        Process PMC open access dataset.

        Args:
            dataset_name: HuggingFace dataset name
            max_records: Maximum number of records to process (None for all)
            num_workers: Number of parallel workers (default: 1 for sequential)
        """
        from datasets import load_dataset

        self.logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)

        # Get already processed PMIDs
        processed = self.get_processed_pmids()
        self.logger.info(f"Already processed: {len(processed)} PMIDs")

        self.logger.info("Filtering dataset...")

        # Single map: add pmc_index and is_valid flag
        def add_metadata(example, idx):
            pmid = int(example.get("pmid", 0))
            example["pmc_index"] = idx
            example["is_valid"] = pmid != 0 and pmid not in processed
            return example

        dataset_with_metadata = dataset["train"].map(
            add_metadata,
            with_indices=True,
            num_proc=min(num_workers, 16),
            desc="Adding metadata"
        )

        filtered_dataset = dataset_with_metadata.filter(
            lambda x: x["is_valid"],
            num_proc=min(num_workers, 16),
            desc="Filtering valid PMIDs"
        )

        self.logger.info(f"Filtered to {len(filtered_dataset)} valid PMIDs")

        if max_records and max_records < len(filtered_dataset):
            filtered_dataset = filtered_dataset.select(range(max_records))
            self.logger.info(f"Limited to first {max_records} records")

        self.logger.info("Extracting PMIDs and PMC indices...")
        pmids = filtered_dataset["pmid"]
        pmc_indices = filtered_dataset["pmc_index"]
        to_process = [{"pmid": pmid, "pmc_index": idx} for pmid, idx in tqdm(zip(pmids, pmc_indices), total=len(pmids), desc="Extracting PMIDs and PMC indices")]
        
        self.stats["skipped"] = len(dataset["train"]) - len(to_process)
        self.logger.info(f"Processing {len(to_process)} PMIDs with {num_workers} worker(s)")

        if num_workers == 1:
            # Sequential processing
            self._process_sequential(to_process)
        else:
            # Parallel processing
            self._process_parallel(to_process, num_workers)

        # Final stats
        self._print_stats()

    def _process_sequential(self, to_process: List[Dict[str, int]]):
        """Process PMIDs sequentially (original behavior)."""
        for item in tqdm(to_process, desc="Fetching metadata"):
            success = self.process_pmid(item["pmid"], item["pmc_index"])

            self.stats["processed"] += 1
            if success:
                self.stats["success"] += 1
            else:
                self.stats["errors"] += 1

            # Save checkpoint every 1000 records
            if self.stats["processed"] % 1000 == 0:
                self._save_checkpoint()

    def _process_parallel(self, to_process: List[Dict[str, int]], num_workers: int):
        """Process PMIDs in parallel with multiple workers."""
        self.logger.info(f"Processing with {num_workers} workers (full speed, no limits)")
        print(f"\nStarting parallel processing of {len(to_process)} PMIDs with {num_workers} workers...")
        print("Initializing workers (loading FastText models)...", flush=True)

        # Create partial function with fixed parameters
        worker_partial = partial(
            _worker_process_pmid,
            db_path=str(self.db_path)
        )

        # Process with pool - initializer loads FastText model once per worker
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=("/mnt/home/al2644/storage/fasttext/models/lid.176.bin",)
        ) as pool:
            print("Workers initialized! Starting metadata fetching...", flush=True)
            results = list(tqdm(
                pool.imap_unordered(worker_partial, to_process, chunksize=1),
                total=len(to_process),
                desc="Fetching metadata (parallel)"
            ))

        # Update stats
        self.stats["processed"] = len(results)
        self.stats["success"] = sum(results)
        self.stats["errors"] = len(results) - sum(results)

        self._save_checkpoint()

    def _save_checkpoint(self):
        """Save progress checkpoint."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats
        }

        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.info(f"Checkpoint saved: {self.stats}")

    def _print_stats(self):
        """Print final statistics."""
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete")
        self.logger.info(f"Processed: {self.stats['processed']}")
        self.logger.info(f"Success: {self.stats['success']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")

        if self.stats['processed'] > 0:
            success_rate = self.stats['success'] / self.stats['processed'] * 100
            self.logger.info(f"Success Rate: {success_rate:.1f}%")

        self.logger.info("=" * 60)

    def query(self, condition: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Query articles with custom WHERE condition.

        Args:
            condition: SQL WHERE clause (without WHERE keyword)
            params: Parameters for the query

        Returns:
            List of article dictionaries

        Example:
            query("citation_count > ? AND language = ?", (50, "eng"))
        """
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"SELECT * FROM pubmedmetadata WHERE {condition}", params)
            rows = cursor.fetchall()

            # Convert to dictionaries and parse JSON fields
            results = []
            for row in rows:
                article = dict(row)

                # Parse JSON fields
                for field in ['article_ids', 'authors', 'mesh_headings', 'keywords', 'article_references']:
                    if article.get(field):
                        try:
                            article[field] = json.loads(article[field])
                        except:
                            article[field] = None

                results.append(article)

            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pubmedmetadata")
            total = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM pubmedmetadata WHERE status = 'success'")
            success = cursor.fetchone()[0]

            cursor = conn.execute("SELECT language, COUNT(*) FROM pubmedmetadata WHERE language != '' GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10")
            languages = dict(cursor.fetchall())

            cursor = conn.execute("SELECT pub_year, COUNT(*) FROM pubmedmetadata WHERE pub_year != '' GROUP BY pub_year ORDER BY pub_year DESC LIMIT 10")
            years = dict(cursor.fetchall())

            return {
                "total": total,
                "success": success,
                "errors": total - success,
                "top_languages": languages,
                "recent_years": years
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PubMed metadata pipeline")
    parser.add_argument("--max", type=int, help="Maximum records to process")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")

    args = parser.parse_args()

    pipeline = SimplePubMedPipeline()

    if args.stats:
        stats = pipeline.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        pipeline.process_dataset(max_records=args.max, num_workers=args.workers)
