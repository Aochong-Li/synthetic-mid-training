#!/usr/bin/env python3
"""
Test script: Fetch metadata for 100 PMIDs and explore the database.

This script will:
1. Create a test database with 100 sample articles
2. Show statistics
3. Run example queries
4. Let you explore the data interactively
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pubmed_simple_pipeline import SimplePubMedPipeline


def main():
    print("=" * 70)
    print("PUBMED PIPELINE TEST".center(70))
    print("=" * 70)
    print("\nThis will fetch metadata for 100 sample articles from PMC.")
    print("Database will be created at: /mnt/home/al2644/storage/pubmed/metadata/pubmed.db")
    print()

    response = input("Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = SimplePubMedPipeline(output_dir="/mnt/home/al2644/storage/pubmed/metadata")

    # Process 100 records
    print("\nFetching metadata for 100 articles...")
    print("This may take 1-2 minutes (rate limited to 3 req/sec)")
    print("\nTip: Use --workers N to speed up with parallel processing")
    print("Example: python test_pipeline.py --workers 4")
    print()

    import sys
    num_workers = 1
    if "--workers" in sys.argv:
        try:
            idx = sys.argv.index("--workers")
            num_workers = int(sys.argv[idx + 1])
            print(f"Using {num_workers} parallel workers\n")
        except:
            pass

    pipeline.process_dataset(max_records=100, num_workers=num_workers)

    # Show statistics
    print("\n" + "=" * 70)
    print("DATABASE STATISTICS".center(70))
    print("=" * 70)

    stats = pipeline.get_stats()
    print(f"\nTotal articles:  {stats['total']}")
    print(f"Successful:      {stats['success']}")
    print(f"Errors:          {stats['errors']}")

    if stats['total'] == 0:
        print("\nNo data was fetched. Check the logs for errors.")
        return

    if stats['top_languages']:
        print("\nLanguages:")
        for lang, count in list(stats['top_languages'].items())[:5]:
            print(f"  {lang}: {count}")

    if stats['recent_years']:
        print("\nPublication Years:")
        for year, count in list(stats['recent_years'].items())[:5]:
            print(f"  {year}: {count}")

    # Show example queries
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES".center(70))
    print("=" * 70)

    print("\n1. Articles with >20 references:")
    articles = pipeline.query("citation_count > 20 AND status = 'success'", ())
    print(f"   Found: {len(articles)}")
    for i, article in enumerate(articles[:3]):
        print(f"\n   {i+1}. [{article['citation_count']} refs] {article['article_title'][:55]}...")

    print("\n2. English articles:")
    articles = pipeline.query("language = 'eng' AND status = 'success'", ())
    print(f"   Found: {len(articles)}")
    for i, article in enumerate(articles[:3]):
        print(f"   {i+1}. {article['article_title'][:60]}...")

    print("\n3. Recent articles (2020+):")
    articles = pipeline.query("pub_year >= '2020' AND status = 'success'", ())
    print(f"   Found: {len(articles)}")
    for i, article in enumerate(articles[:3]):
        print(f"   {i+1}. [{article['pub_year']}] {article['article_title'][:55]}...")

    # Interactive mode
    print("\n" + "=" * 70)
    print("EXPLORE THE DATABASE".center(70))
    print("=" * 70)
    print("\nYou can now explore the database using Python:")
    print()
    print("  from pubmed_simple_pipeline import SimplePubMedPipeline")
    print("  pipeline = SimplePubMedPipeline()")
    print()
    print("  # Query examples:")
    print("  articles = pipeline.query('citation_count > 30')")
    print("  articles = pipeline.query('language = \"eng\"')")
    print("  articles = pipeline.query('pub_year = \"2023\"')")
    print("  articles = pipeline.query('journal_title LIKE \"%Nature%\"')")
    print()
    print("Or use the query utility:")
    print("  python query_pubmed.py --stats")
    print("  python query_pubmed.py --examples")
    print("  python query_pubmed.py --interactive")
    print()
    print("Database location:")
    print(f"  {pipeline.db_path}")
    print()
    print("=" * 70)
    print("\nTest complete!")


if __name__ == "__main__":
    main()
