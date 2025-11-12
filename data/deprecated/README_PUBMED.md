# PubMed Metadata Pipeline

Simple pipeline to fetch PubMed metadata and store in SQLite database.

## Quick Start

### 1. Test with 100 samples (recommended first step)

```bash
cd /mnt/home/al2644/research/projects/synthetic-mid-training/data
python test_pipeline.py
```

This will:
- Fetch metadata for 100 articles
- Create database at `/mnt/home/al2644/storage/pubmed/metadata/pubmed.db`
- Show example queries
- Take ~1-2 minutes

### 2. Run full pipeline

```bash
# Process all articles (will take many hours)
python pubmed_simple_pipeline.py

# Process specific number
python pubmed_simple_pipeline.py --max 1000

# Use parallel workers (faster!)
python pubmed_simple_pipeline.py --max 1000 --workers 4

# Test script with parallel workers
python test_pipeline.py --workers 4
```

### 3. Query the database

```bash
# Show statistics
python query_pubmed.py --stats

# Run example queries
python query_pubmed.py --examples

# Interactive query mode
python query_pubmed.py --interactive
```

## Files

- `pubmed_simple_pipeline.py` - Main pipeline (all-in-one)
- `test_pipeline.py` - Test script (100 samples)
- `query_pubmed.py` - Query utility
- `filter_pubmed.py` - PubMedProcessor class (your existing code)

## Database Schema

```sql
CREATE TABLE articles (
    pmid INTEGER PRIMARY KEY,

    -- Basic info
    article_title TEXT,
    abstract TEXT,
    language TEXT,

    -- Journal
    journal_title TEXT,
    journal_issn TEXT,

    -- Date
    pub_year TEXT,
    pub_month TEXT,
    pub_day TEXT,

    -- JSON fields
    article_ids TEXT,      -- DOI, PMC, etc.
    authors TEXT,          -- Author list
    mesh_headings TEXT,    -- MeSH terms
    keywords TEXT,         -- Keywords
    article_references TEXT,       -- References

    -- Metadata
    citation_count INTEGER,
    pmc_index INTEGER,
    fetched_at TIMESTAMP,
    status TEXT,
    error_message TEXT
)
```

## Query Examples

### Python API

```python
from pubmed_simple_pipeline import SimplePubMedPipeline

pipeline = SimplePubMedPipeline()

# Get articles with many references
articles = pipeline.query("citation_count > 50")

# Filter by language and year
articles = pipeline.query(
    "language = ? AND pub_year = ?",
    ("eng", "2023")
)

# Complex query
articles = pipeline.query(
    "pub_year >= '2020' AND citation_count > 30 AND language = 'eng'"
)

# Search in title or journal
articles = pipeline.query("journal_title LIKE '%Nature%'")

# Get statistics
stats = pipeline.get_stats()
```

### SQL Examples

```sql
-- High citation count
SELECT * FROM articles WHERE citation_count > 50;

-- English articles from 2023
SELECT * FROM articles WHERE language = 'eng' AND pub_year = '2023';

-- Nature journals
SELECT * FROM articles WHERE journal_title LIKE '%Nature%';

-- Recent high-impact
SELECT * FROM articles
WHERE pub_year >= '2020' AND citation_count > 30 AND language = 'eng';

-- Count by year
SELECT pub_year, COUNT(*) FROM articles
GROUP BY pub_year ORDER BY pub_year DESC;
```

## Features

- **Parallel processing**: Use `--workers N` for N parallel workers (faster!)
- **Auto-retry**: Network/SSL errors retry up to 3 times with backoff (1s, 5s, 30s)
- **No rate limiting**: Maximum speed (be aware of potential NCBI rate limits)
- **Resume**: Skips already-processed PMIDs
- **Checkpoints**: Saves progress every 1000 records
- **Logging**: Detailed logs in output directory
- **JSON fields**: Authors, MeSH terms, keywords stored as JSON

## Output Structure

```
/mnt/home/al2644/storage/pubmed/metadata/
├── pubmed.db                        # SQLite database
├── checkpoint.json                  # Progress checkpoint
└── pipeline_YYYYMMDD.log           # Daily log file
```

## Error Handling

- **Retryable errors**: SSL, connection, timeout, rate limit → auto-retry
- **Non-retryable errors**: 404, invalid PMID → logged and skipped
- PMIDs with `pmid=0` are automatically skipped

## Performance

**Sequential (1 worker):**
- Rate: As fast as network allows
- 100 articles: ~30 seconds
- 1000 articles: ~5 minutes

**Parallel (64+ workers):**
- Rate: Limited only by network bandwidth and CPU
- Significant speedup with many workers
- Recommended: 16-64 workers depending on your machine

**Full dataset:** Much faster without rate limiting (hours instead of days)

## Tips

1. **Start with test**: Run `test_pipeline.py` first to verify everything works
2. **Monitor progress**: Check logs or run with `--stats` periodically
3. **Resume after crash**: Just run again, it skips processed PMIDs
4. **Query efficiently**: Use indexed columns (language, pub_year, journal_title, citation_count)
5. **JSON queries**: For complex JSON queries, extract to indexed columns first

## Troubleshooting

**Database locked:**
- Close other connections to the database

**No data fetched:**
- Check logs in `/mnt/home/al2644/storage/pubmed/metadata/`
- Verify FastText model path is correct

**Memory issues:**
- Limit results: `SELECT * FROM articles LIMIT 1000`
- Use pagination with OFFSET

## Direct Database Access

```bash
# SQLite command line
sqlite3 /mnt/home/al2644/storage/pubmed/metadata/pubmed.db

# Example queries
sqlite> SELECT COUNT(*) FROM articles;
sqlite> SELECT language, COUNT(*) FROM articles GROUP BY language;
sqlite> SELECT * FROM articles WHERE citation_count > 50 LIMIT 5;
```

## Next Steps

After testing with 100 samples:
1. Run full pipeline: `python pubmed_simple_pipeline.py`
2. Query your data: `python query_pubmed.py --interactive`
3. Build analysis scripts using the query API
