#!/usr/bin/env python3
"""
Parse PubMed baseline XML files and store metadata in SQLite database.
Uses producer-consumer pattern: workers parse XML, single writer inserts to DB.
This avoids database lock contention.
"""

import sqlite3
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
from queue import Empty
from tqdm import tqdm
import argparse
import logging
import time


class PubMedXMLParser:
    """Parser for PubMed XML files."""

    @staticmethod
    def _text(el: Optional[ET.Element]) -> str:
        """Extract text from XML element."""
        return (el.text or "").strip() if el is not None else ""

    @staticmethod
    def _find(root: ET.Element, path: str) -> Optional[ET.Element]:
        """Find single element."""
        return root.find(path)

    @staticmethod
    def _findall(root: ET.Element, path: str) -> List[ET.Element]:
        """Find all matching elements."""
        return root.findall(path)

    def parse_article_title(self, article: ET.Element) -> str:
        """Extract article title."""
        return self._text(self._find(article, "./MedlineCitation/Article/ArticleTitle"))

    def parse_journal_title(self, article: ET.Element) -> str:
        """Extract journal title."""
        return self._text(self._find(article, "./MedlineCitation/Article/Journal/Title"))

    def parse_journal_issn(self, article: ET.Element) -> str:
        """Extract journal ISSN."""
        issn_elem = self._find(article, "./MedlineCitation/Article/Journal/ISSN")
        return self._text(issn_elem) if issn_elem is not None else ""

    def parse_journal_metadata(self, article: ET.Element) -> Dict[str, str]:
        """Extract journal volume and issue."""
        journal_issue = self._find(article, "./MedlineCitation/Article/Journal/JournalIssue")
        volume = self._text(self._find(journal_issue, "./Volume")) if journal_issue is not None else ""
        issue = self._text(self._find(journal_issue, "./Issue")) if journal_issue is not None else ""
        return {"volume": volume, "issue": issue}

    def parse_publication_date(self, article: ET.Element) -> Dict[str, str]:
        """Extract publication date."""
        pubdate = self._find(article, "./MedlineCitation/Article/Journal/JournalIssue/PubDate")
        if pubdate is None:
            return {"year": "", "month": "", "day": ""}

        return {
            "year": self._text(self._find(pubdate, "./Year")),
            "month": self._text(self._find(pubdate, "./Month")),
            "day": self._text(self._find(pubdate, "./Day"))
        }

    def parse_article_date(self, article: ET.Element) -> str:
        """Extract electronic publication date as ISO string."""
        article_date = self._find(article, "./MedlineCitation/Article/ArticleDate")
        if article_date is None:
            return ""

        year = self._text(self._find(article_date, "./Year"))
        month = self._text(self._find(article_date, "./Month"))
        day = self._text(self._find(article_date, "./Day"))

        if year and month and day:
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return ""

    def parse_language(self, article: ET.Element) -> str:
        """Extract article language code."""
        return self._text(self._find(article, "./MedlineCitation/Article/Language"))

    def parse_abstract(self, article: ET.Element) -> str:
        """Extract full abstract text."""
        abstract_texts = []
        for ab in self._findall(article, "./MedlineCitation/Article/Abstract/AbstractText"):
            text = "".join(ab.itertext()).strip()
            if text:
                abstract_texts.append(text)

        return "\n\n".join(abstract_texts) if abstract_texts else ""

    def parse_authors(self, article: ET.Element) -> List[Dict[str, Any]]:
        """Extract author information."""
        authors = []
        for au in self._findall(article, "./MedlineCitation/Article/AuthorList/Author"):
            last = self._text(self._find(au, "./LastName"))
            fore = self._text(self._find(au, "./ForeName"))
            initials = self._text(self._find(au, "./Initials"))
            collective = self._text(self._find(au, "./CollectiveName"))

            # Get affiliations
            affs = [self._text(self._find(a, "./Affiliation"))
                   for a in self._findall(au, "./AffiliationInfo")]
            affs = [a for a in affs if a]

            # Get identifiers (e.g., ORCID)
            identifiers = []
            for ident in self._findall(au, "./Identifier"):
                identifiers.append({
                    "value": self._text(ident),
                    "source": ident.get("Source", "")
                })

            authors.append({
                "last": last,
                "fore": fore,
                "initials": initials,
                "collective_name": collective,
                "affiliations": affs,
                "identifiers": identifiers
            })

        return authors

    def parse_mesh_headings(self, article: ET.Element) -> List[Dict[str, Any]]:
        """Extract MeSH terms."""
        mesh_list = []
        for mesh in self._findall(article, "./MedlineCitation/MeshHeadingList/MeshHeading"):
            descriptor_elem = self._find(mesh, "./DescriptorName")
            if descriptor_elem is not None:
                descriptor = {
                    "name": self._text(descriptor_elem),
                    "ui": descriptor_elem.get("UI", ""),
                    "major_topic": descriptor_elem.get("MajorTopicYN", "N")
                }

                # Get qualifiers
                qualifiers = []
                for qual in self._findall(mesh, "./QualifierName"):
                    qualifiers.append({
                        "name": self._text(qual),
                        "ui": qual.get("UI", ""),
                        "major_topic": qual.get("MajorTopicYN", "N")
                    })

                mesh_list.append({
                    "descriptor": descriptor,
                    "qualifiers": qualifiers
                })

        return mesh_list

    def parse_keywords(self, article: ET.Element) -> List[Dict[str, str]]:
        """Extract keywords."""
        keywords = []
        for kw in self._findall(article, "./MedlineCitation/KeywordList/Keyword"):
            keywords.append({
                "keyword": self._text(kw),
                "major_topic": kw.get("MajorTopicYN", "N")
            })
        return keywords

    def parse_article_ids(self, article: ET.Element) -> Dict[str, str]:
        """Extract article IDs (pubmed, pmc, doi, etc)."""
        ids = {}
        for aid in self._findall(article, "./PubmedData/ArticleIdList/ArticleId"):
            id_type = (aid.get("IdType") or "").strip().lower()
            value = self._text(aid)
            if id_type and value:
                ids[id_type] = value

        # Ensure PMID exists
        if "pubmed" not in ids:
            pmid_val = self._text(self._find(article, "./MedlineCitation/PMID"))
            if pmid_val:
                ids["pubmed"] = pmid_val

        return ids

    def parse_references(self, article: ET.Element) -> List[Dict[str, Any]]:
        """Extract references."""
        refs = []
        for ref in self._findall(article, "./PubmedData/ReferenceList/Reference"):
            citation = self._text(self._find(ref, "./Citation"))
            ids = []
            for rid in self._findall(ref, "./ArticleIdList/ArticleId"):
                ids.append({
                    "type": rid.get("IdType") or "",
                    "value": self._text(rid)
                })
            refs.append({"citation": citation, "ids": ids})

        return refs

    def parse_article(self, article_elem: ET.Element, source_file: str) -> Dict[str, Any]:
        """Parse a single PubmedArticle element into a record."""
        # Get PMID
        pmid_elem = self._find(article_elem, "./MedlineCitation/PMID")
        pmid = int(self._text(pmid_elem)) if pmid_elem is not None else 0

        if pmid == 0:
            return None

        # Parse all fields
        pub_date = self.parse_publication_date(article_elem)
        journal_meta = self.parse_journal_metadata(article_elem)
        authors = self.parse_authors(article_elem)
        mesh = self.parse_mesh_headings(article_elem)
        keywords = self.parse_keywords(article_elem)
        article_ids = self.parse_article_ids(article_elem)
        references = self.parse_references(article_elem)

        record = {
            "pmid": pmid,
            "article_title": self.parse_article_title(article_elem),
            "abstract": self.parse_abstract(article_elem),
            "language": self.parse_language(article_elem),
            "journal_title": self.parse_journal_title(article_elem),
            "journal_issn": self.parse_journal_issn(article_elem),
            "journal_volume": journal_meta["volume"],
            "journal_issue": journal_meta["issue"],
            "pub_year": pub_date["year"],
            "pub_month": pub_date["month"],
            "pub_day": pub_date["day"],
            "article_date": self.parse_article_date(article_elem),
            "article_ids": json.dumps(article_ids),
            "authors": json.dumps(authors),
            "mesh_headings": json.dumps(mesh),
            "keywords": json.dumps(keywords),
            "article_references": json.dumps(references),
            "citation_count": len(references),
            "source_file": source_file,
            "processed_at": datetime.now().isoformat()
        }

        return record

    def parse_xml_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a PubMed XML file and return list of article records."""
        records = []

        # Parse XML incrementally to handle large files
        context = ET.iterparse(file_path, events=('end',))

        for event, elem in context:
            if elem.tag == 'PubmedArticle':
                try:
                    record = self.parse_article(elem, Path(file_path).name)
                    if record:
                        records.append(record)
                except Exception as e:
                    logging.error(f"Error parsing article in {file_path}: {e}")
                finally:
                    # Clear element to free memory
                    elem.clear()

        return records


class PubMedDatabase:
    """SQLite database manager for PubMed metadata."""

    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            # Enable WAL mode for concurrent writes
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=100000")
            conn.execute("PRAGMA temp_store=MEMORY")

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
                    journal_volume TEXT,
                    journal_issue TEXT,

                    -- Dates
                    pub_year TEXT,
                    pub_month TEXT,
                    pub_day TEXT,
                    article_date TEXT,

                    -- JSON fields
                    article_ids TEXT,
                    authors TEXT,
                    mesh_headings TEXT,
                    keywords TEXT,
                    article_references TEXT,

                    -- Metadata
                    citation_count INTEGER DEFAULT 0,
                    source_file TEXT,
                    processed_at TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON pubmedmetadata(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON pubmedmetadata(pub_year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal ON pubmedmetadata(journal_title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_citations ON pubmedmetadata(citation_count)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_file ON pubmedmetadata(source_file)")

            conn.commit()

    def batch_insert_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Batch insert articles into database. Returns number of inserted articles."""
        if not articles:
            return 0

        inserted = 0
        with sqlite3.connect(self.db_path, timeout=600.0) as conn:
            # Use executemany for batch insert
            try:
                data = [
                    (
                        article["pmid"],
                        article["article_title"],
                        article["abstract"],
                        article["language"],
                        article["journal_title"],
                        article["journal_issn"],
                        article["journal_volume"],
                        article["journal_issue"],
                        article["pub_year"],
                        article["pub_month"],
                        article["pub_day"],
                        article["article_date"],
                        article["article_ids"],
                        article["authors"],
                        article["mesh_headings"],
                        article["keywords"],
                        article["article_references"],
                        article["citation_count"],
                        article["source_file"],
                        article["processed_at"]
                    )
                    for article in articles
                ]

                conn.executemany("""
                    INSERT OR REPLACE INTO pubmedmetadata (
                        pmid, article_title, abstract, language,
                        journal_title, journal_issn, journal_volume, journal_issue,
                        pub_year, pub_month, pub_day, article_date,
                        article_ids, authors, mesh_headings, keywords, article_references,
                        citation_count, source_file, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, data)

                inserted = len(data)
                conn.commit()
            except Exception as e:
                logging.error(f"Error batch inserting {len(articles)} articles: {e}")

        return inserted

    def get_processed_files(self) -> set:
        """Get set of already processed source files."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            cursor = conn.execute("SELECT DISTINCT source_file FROM pubmedmetadata")
            return {row[0] for row in cursor.fetchall()}

    def count_articles(self) -> int:
        """Count total articles in database."""
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pubmedmetadata")
            return cursor.fetchone()[0]


def parse_file_worker(file_path: str) -> tuple:
    """Worker function to parse a single XML file."""
    parser = PubMedXMLParser()

    try:
        # Parse XML file
        articles = parser.parse_xml_file(file_path)
        return True, Path(file_path).name, articles
    except Exception as e:
        return False, Path(file_path).name, str(e)


def database_writer_process(queue, db_path: str, total_files: int):
    """Dedicated process for writing to database."""
    db = PubMedDatabase(db_path)

    processed_files = 0
    total_articles = 0
    total_inserted = 0

    batch = []
    batch_size = 5000  # Insert every 5000 articles

    pbar = tqdm(total=total_files, desc="Processing files", position=0)

    while True:
        try:
            # Get item from queue (timeout to check for completion)
            item = queue.get(timeout=5)

            if item is None:  # Poison pill to signal completion
                break

            success, filename, data = item

            if success:
                # Add articles to batch
                batch.extend(data)

                # Insert if batch is large enough
                if len(batch) >= batch_size:
                    inserted = db.batch_insert_articles(batch)
                    total_inserted += inserted
                    total_articles += len(batch)
                    batch = []

                processed_files += 1
                pbar.update(1)
                pbar.set_postfix({
                    'articles': total_articles,
                    'inserted': total_inserted,
                    'batch': len(batch)
                })
            else:
                logging.error(f"Failed to parse {filename}: {data}")
                processed_files += 1
                pbar.update(1)

        except Empty:
            # Check if there's a batch waiting
            if batch:
                inserted = db.batch_insert_articles(batch)
                total_inserted += inserted
                total_articles += len(batch)
                pbar.set_postfix({
                    'articles': total_articles,
                    'inserted': total_inserted,
                    'batch': 0
                })
                batch = []
            continue

    # Insert remaining batch
    if batch:
        inserted = db.batch_insert_articles(batch)
        total_inserted += inserted
        total_articles += len(batch)

    pbar.close()

    print(f"\n{'='*60}")
    print(f"Database Writer Summary:")
    print(f"  Files processed: {processed_files}")
    print(f"  Articles parsed: {total_articles}")
    print(f"  Articles inserted: {total_inserted}")
    print(f"{'='*60}")


def process_files_parallel(xml_dir: str, db_path: str, num_workers: int = 8, skip_processed: bool = True):
    """Process PubMed XML files in parallel with dedicated database writer."""
    xml_dir = Path(xml_dir)
    xml_files = sorted(xml_dir.glob("pubmed25n*.xml"))

    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return

    print(f"Found {len(xml_files)} XML files")

    # Get already processed files
    db = PubMedDatabase(db_path)
    processed_files = db.get_processed_files() if skip_processed else set()

    if processed_files:
        print(f"Skipping {len(processed_files)} already processed files")
        xml_files = [f for f in xml_files if f.name not in processed_files]

    if not xml_files:
        print("No new files to process!")
        return

    print(f"Processing {len(xml_files)} files with {num_workers} parser workers + 1 database writer")

    # Create queue for communication
    manager = Manager()
    queue = manager.Queue(maxsize=num_workers * 2)  # Limit queue size

    # Start database writer process
    writer = Process(target=database_writer_process, args=(queue, db_path, len(xml_files)))
    writer.start()

    # Process files with worker pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(parse_file_worker, str(f)): f
            for f in xml_files
        }

        # Collect results and put in queue
        for future in as_completed(futures):
            try:
                result = future.result()
                queue.put(result)
            except Exception as e:
                file_path = futures[future]
                logging.error(f"Exception processing {file_path.name}: {e}")
                queue.put((False, file_path.name, str(e)))

    # Signal writer to finish
    queue.put(None)

    # Wait for writer to finish
    writer.join()

    # Final stats
    total_in_db = db.count_articles()
    print(f"\nTotal articles in database: {total_in_db:,}")


def main():
    parser = argparse.ArgumentParser(description="Parse PubMed baseline XML files to SQLite")
    parser.add_argument("--xml-dir", type=str, default="/mnt/home/al2644/storage/pubmed/baseline",
                        help="Directory containing PubMed XML files")
    parser.add_argument("--db-path", type=str, default="/mnt/home/al2644/storage/pubmed/db/pubmed.db",
                        help="Path to SQLite database")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel parser workers")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip already processed files")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    process_files_parallel(
        args.xml_dir,
        args.db_path,
        num_workers=args.workers,
        skip_processed=not args.no_skip
    )


if __name__ == "__main__":
    main()
