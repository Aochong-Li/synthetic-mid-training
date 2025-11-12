#!/usr/bin/env python3
"""
Test script to parse 50 PubMed articles and insert into database.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from parse_pubmed_baseline import PubMedXMLParser, PubMedDatabase


def parse_limited_articles(xml_file: str, max_articles: int = 50):
    """Parse limited number of articles from XML file."""
    parser = PubMedXMLParser()
    articles = []

    print(f"Parsing up to {max_articles} articles from {Path(xml_file).name}...")

    # Parse XML incrementally
    context = ET.iterparse(xml_file, events=('end',))

    for event, elem in context:
        if elem.tag == 'PubmedArticle':
            try:
                record = parser.parse_article(elem, Path(xml_file).name)
                if record:
                    articles.append(record)
                    if len(articles) >= max_articles:
                        break
            except Exception as e:
                print(f"Error parsing article: {e}")
            finally:
                elem.clear()

    print(f"Successfully parsed {len(articles)} articles")
    return articles


def test_parser(xml_dir: str, db_path: str, num_articles: int = 50):
    """Test parsing and database insertion."""
    # Find first XML file
    xml_dir = Path(xml_dir)
    xml_files = sorted(xml_dir.glob("pubmed25n*.xml"))

    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return

    first_file = xml_files[0]
    print(f"Testing with file: {first_file}")

    # Parse articles
    articles = parse_limited_articles(str(first_file), num_articles)

    if not articles:
        print("No articles parsed!")
        return

    # Show sample article
    print("\n" + "=" * 60)
    print("Sample Article:")
    print("=" * 60)
    sample = articles[0]
    print(f"PMID: {sample['pmid']}")
    print(f"Title: {sample['article_title']}")
    print(f"Journal: {sample['journal_title']}")
    print(f"Language: {sample['language']}")
    print(f"Year: {sample['pub_year']}")
    print(f"Citation Count: {sample['citation_count']}")
    print(f"Abstract: {sample['abstract'][:200]}..." if sample['abstract'] else "Abstract: None")

    # Insert into database
    print("\n" + "=" * 60)
    print(f"Inserting {len(articles)} articles into database...")
    print("=" * 60)

    db = PubMedDatabase(db_path)
    inserted = db.insert_articles(articles)

    print(f"Successfully inserted: {inserted}/{len(articles)} articles")

    # Verify database
    total_count = db.count_articles()
    print(f"Total articles in database: {total_count}")

    # Query and display some records
    print("\n" + "=" * 60)
    print("Sample Database Query (first 5 articles):")
    print("=" * 60)

    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT pmid, article_title, journal_title, pub_year, citation_count
            FROM pubmedmetadata
            LIMIT 5
        """)
        for row in cursor:
            pmid, title, journal, year, citations = row
            print(f"\nPMID: {pmid}")
            print(f"  Title: {title[:80]}...")
            print(f"  Journal: {journal}")
            print(f"  Year: {year}")
            print(f"  Citations: {citations}")

    # Show statistics
    print("\n" + "=" * 60)
    print("Database Statistics:")
    print("=" * 60)

    with sqlite3.connect(db_path) as conn:
        # Language distribution
        cursor = conn.execute("""
            SELECT language, COUNT(*) as count
            FROM pubmedmetadata
            WHERE language != ''
            GROUP BY language
            ORDER BY count DESC
            LIMIT 5
        """)
        print("\nTop Languages:")
        for lang, count in cursor:
            print(f"  {lang}: {count}")

        # Year distribution
        cursor = conn.execute("""
            SELECT pub_year, COUNT(*) as count
            FROM pubmedmetadata
            WHERE pub_year != ''
            GROUP BY pub_year
            ORDER BY pub_year DESC
            LIMIT 5
        """)
        print("\nRecent Years:")
        for year, count in cursor:
            print(f"  {year}: {count}")

        # Citation statistics
        cursor = conn.execute("""
            SELECT
                AVG(citation_count) as avg_citations,
                MAX(citation_count) as max_citations,
                MIN(citation_count) as min_citations
            FROM pubmedmetadata
        """)
        avg, max_c, min_c = cursor.fetchone()
        print(f"\nCitation Statistics:")
        print(f"  Average: {avg:.2f}")
        print(f"  Maximum: {max_c}")
        print(f"  Minimum: {min_c}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PubMed parser with limited articles")
    parser.add_argument("--xml-dir", type=str, default="/mnt/home/al2644/storage/pubmed/baseline",
                        help="Directory containing PubMed XML files")
    parser.add_argument("--db-path", type=str, default="/mnt/home/al2644/storage/pubmed/db/pubmed_test.db",
                        help="Path to test SQLite database")
    parser.add_argument("--num-articles", type=int, default=50,
                        help="Number of articles to parse and insert")

    args = parser.parse_args()

    test_parser(args.xml_dir, args.db_path, args.num_articles)
