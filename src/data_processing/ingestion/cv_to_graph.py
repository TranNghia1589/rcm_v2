from __future__ import annotations

# This script reuses the same ETL pipeline as jobs_to_neo4j.py
# because CV nodes/links are synced from PostgreSQL tables.

from src.data_processing.ingestion.jobs_to_neo4j import main


if __name__ == "__main__":
    main()

