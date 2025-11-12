#!/usr/bin/env python3
"""
Simple query utility for PubMed database.
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pubmed_simple_pipeline import SimplePubMedPipeline


def show_stats(pipeline):
    """Display database statistics."""
    stats = pipeline.get_stats()

    print("\n" + "=" * 70)
    print("DATABASE STATISTICS".center(70))
    print("=" * 70)
    print(f"Total articles:  {stats['total']:,}")
    print(f"Successful:      {stats['success']:,}")
    print(f"Errors:          {stats['errors']:,}")

    if stats['top_languages']:
        print("\nTop Languages:")
        for lang, count in stats['top_languages'].items():
            print(f"  {lang:10s} {count:,}")

    if stats['recent_years']:
        print("\nRecent Years:")
        for year, count in stats['recent_years'].items():
            print(f"  {year:10s} {count:,}")

    print("=" * 70 + "\n")


def run_examples(pipeline):
    """Run example queries."""
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES".center(70))
    print("=" * 70)

    # Example 1: Articles with many citations
    print("\n1. Articles with > 50 references:")
    print("-" * 70)
    articles = pipeline.query("citation_count > ? AND status = 'success'", (50,))
    print(f"Found {len(articles)} articles")
    for i, article in enumerate(articles[:3]):
        print(f"\n  {i+1}. [{article['citation_count']} refs] {article['article_title'][:60]}...")
        print(f"     Journal: {article['journal_title']}")
        print(f"     Year: {article['pub_year']}")

    # Example 2: English articles from 2023
    print("\n2. English articles from 2023:")
    print("-" * 70)
    articles = pipeline.query("language = 'eng' AND pub_year = '2023'", ())
    print(f"Found {len(articles)} articles")
    for i, article in enumerate(articles[:3]):
        print(f"\n  {i+1}. {article['article_title'][:70]}...")

    # Example 3: Articles from Nature journals
    print("\n3. Articles from Nature journals:")
    print("-" * 70)
    articles = pipeline.query("journal_title LIKE '%Nature%' AND status = 'success'", ())
    print(f"Found {len(articles)} articles")

    journals = {}
    for article in articles:
        j = article['journal_title']
        journals[j] = journals.get(j, 0) + 1

    for journal, count in sorted(journals.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {journal}: {count}")

    # Example 4: Recent high-impact articles
    print("\n4. Recent articles (2022+) with >30 references:")
    print("-" * 70)
    articles = pipeline.query(
        "pub_year IN ('2022', '2023', '2024') AND citation_count > 30 AND status = 'success'",
        ()
    )
    print(f"Found {len(articles)} articles")

    # Sort by citations
    articles.sort(key=lambda x: x['citation_count'], reverse=True)
    for i, article in enumerate(articles[:3]):
        print(f"\n  {i+1}. [{article['citation_count']} refs, {article['pub_year']}]")
        print(f"     {article['article_title'][:65]}...")

    print("\n" + "=" * 70 + "\n")


def custom_query(pipeline):
    """Interactive custom query interface."""
    print("\n" + "=" * 70)
    print("CUSTOM QUERY INTERFACE".center(70))
    print("=" * 70)
    print("\nEnter SQL WHERE clause (without 'WHERE')")
    print("Example: citation_count > 50 AND language = 'eng'")
    print("Type 'exit' to quit\n")

    while True:
        try:
            condition = input("Query> ").strip()

            if condition.lower() == 'exit':
                break

            if not condition:
                continue

            articles = pipeline.query(condition)
            print(f"\nFound {len(articles)} articles\n")

            if articles:
                for i, article in enumerate(articles[:10]):
                    print(f"{i+1}. [PMID: {article['pmid']}] {article['article_title'][:60]}...")
                    print(f"   Journal: {article['journal_title']}, Year: {article['pub_year']}")
                    print()

                if len(articles) > 10:
                    print(f"... and {len(articles) - 10} more results\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Check your SQL syntax.\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query PubMed database")
    parser.add_argument("--db", default="/mnt/home/al2644/storage/pubmed/metadata",
                        help="Database directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics only")
    parser.add_argument("--examples", action="store_true",
                        help="Run example queries")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive query mode")

    args = parser.parse_args()

    pipeline = SimplePubMedPipeline(output_dir=args.db)

    if args.stats:
        show_stats(pipeline)
    elif args.examples:
        run_examples(pipeline)
    elif args.interactive:
        custom_query(pipeline)
    else:
        # Default: show stats and examples
        show_stats(pipeline)
        run_examples(pipeline)


if __name__ == "__main__":
    main()
