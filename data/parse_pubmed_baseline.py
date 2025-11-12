#!/usr/bin/env python3
"""
Parse PubMed baseline XML files and store metadata in SQLite database.
Processes files in parallel with configurable number of workers.
"""

import sqlite3
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import logging


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

    def insert_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Insert articles into database. Returns number of inserted articles."""
        if not articles:
            return 0

        inserted = 0
        with sqlite3.connect(self.db_path, timeout=300.0) as conn:
            for article in articles:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO pubmedmetadata (
                            pmid, article_title, abstract, language,
                            journal_title, journal_issn, journal_volume, journal_issue,
                            pub_year, pub_month, pub_day, article_date,
                            article_ids, authors, mesh_headings, keywords, article_references,
                            citation_count, source_file, processed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
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
                    ))
                    inserted += 1
                except Exception as e:
                    logging.error(f"Error inserting PMID {article.get('pmid')}: {e}")

            conn.commit()

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


def process_file_worker(file_path: str, db_path: str) -> tuple:
    """Worker function to process a single XML file."""
    parser = PubMedXMLParser()
    db = PubMedDatabase(db_path)

    try:
        # Parse XML file
        articles = parser.parse_xml_file(file_path)

        # Insert into database
        inserted = db.insert_articles(articles)

        return True, Path(file_path).name, len(articles), inserted
    except Exception as e:
        return False, Path(file_path).name, 0, str(e)


def process_files_parallel(xml_dir: str, db_path: str, num_workers: int = 4, skip_processed: bool = True):
    """Process PubMed XML files in parallel."""
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

    print(f"Processing {len(xml_files)} files with {num_workers} workers")

    # Process files in parallel
    total_articles = 0
    total_inserted = 0
    failed_files = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_file_worker, str(f), db_path): f
            for f in xml_files
        }

        # Process results with progress bar
        with tqdm(total=len(xml_files), desc="Processing files") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    success, filename, count, result = future.result()
                    if success:
                        total_articles += count
                        total_inserted += result
                        pbar.set_postfix({
                            'articles': total_articles,
                            'inserted': total_inserted
                        })
                    else:
                        failed_files.append((filename, result))
                        print(f"\nFailed: {filename} - {result}")
                except Exception as e:
                    failed_files.append((file_path.name, str(e)))
                    print(f"\nException processing {file_path.name}: {e}")
                finally:
                    pbar.update(1)

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete")
    print(f"Total articles parsed: {total_articles:,}")
    print(f"Total articles inserted: {total_inserted:,}")
    print(f"Failed files: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")

    # Database stats
    total_in_db = db.count_articles()
    print(f"\nTotal articles in database: {total_in_db:,}")
    print("=" * 60)


def main():
    """
    python parse_pubmed_baseline.py --xml-dir /mnt/home/al2644/storage/pubmed/baseline --db-path /mnt/home/al2644/storage/pubmed/db/pubmed.db --workers 64 --no-skip
    """
    parser = argparse.ArgumentParser(description="Parse PubMed baseline XML files to SQLite")
    parser.add_argument("--xml-dir", type=str, default="/mnt/home/al2644/storage/pubmed/baseline",
                        help="Directory containing PubMed XML files")
    parser.add_argument("--db-path", type=str, default="/mnt/home/al2644/storage/pubmed/db/pubmed.db",
                        help="Path to SQLite database")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
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
