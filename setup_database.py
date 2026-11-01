"""
setup_database.py
-----------------
One-shot script to:
  1. Create the hospital_readmission database
  2. Apply the full schema (all 4 schemas + 9 tables + 5 views)

Run ONCE after PostgreSQL is installed:
    python setup_database.py

Prerequisites:
  - PostgreSQL installed and running on localhost:5432
  - .env file has POSTGRES_PASSWORD set correctly

What it does:
  - Connects to the default 'postgres' database first
  - Creates 'hospital_readmission' database if it doesn't exist
  - Runs src/database/schema.sql to create all schemas and tables
"""

import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def get_pg_credentials():
    return {
        "host":     os.getenv("POSTGRES_HOST", "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", "5432")),
        "user":     os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "dbname":   os.getenv("POSTGRES_DB", "hospital_readmission"),
    }


def create_database(creds: dict) -> bool:
    """Connect to 'postgres' default DB and create the project database."""
    print(f"Connecting to PostgreSQL at {creds['host']}:{creds['port']} as {creds['user']}...")
    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        user=creds["user"],
        password=creds["password"],
        dbname="postgres",          # connect to default DB first
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Check if DB already exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (creds["dbname"],))
    exists = cursor.fetchone()

    if exists:
        print(f"  Database '{creds['dbname']}' already exists — skipping creation.")
    else:
        cursor.execute(f'CREATE DATABASE "{creds["dbname"]}"')
        print(f"  Database '{creds['dbname']}' created successfully.")

    cursor.close()
    conn.close()
    return True


def apply_schema(creds: dict) -> bool:
    """Apply schema.sql to the project database."""
    schema_file = PROJECT_ROOT / "src" / "database" / "schema.sql"
    if not schema_file.exists():
        print(f"ERROR: Schema file not found: {schema_file}")
        return False

    print(f"Applying schema from {schema_file}...")
    sql = schema_file.read_text(encoding="utf-8")

    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        user=creds["user"],
        password=creds["password"],
        dbname=creds["dbname"],
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Execute the full schema SQL
    cursor.execute(sql)
    print("  Schema applied successfully.")

    # Verify tables were created
    cursor.execute("""
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE schemaname IN ('raw', 'staging', 'curated', 'analytics')
        ORDER BY schemaname, tablename
    """)
    tables = cursor.fetchall()
    print(f"\n  Tables created ({len(tables)}):")
    for schema, table in tables:
        print(f"    {schema}.{table}")

    # Verify views
    cursor.execute("""
        SELECT schemaname, viewname
        FROM pg_views
        WHERE schemaname = 'analytics'
    """)
    views = cursor.fetchall()
    print(f"\n  Views created ({len(views)}):")
    for schema, view in views:
        print(f"    {schema}.{view}")

    cursor.close()
    conn.close()
    return True


def test_connection(creds: dict) -> bool:
    """Quick connectivity test."""
    try:
        conn = psycopg2.connect(
            host=creds["host"],
            port=creds["port"],
            user=creds["user"],
            password=creds["password"],
            dbname="postgres",
        )
        conn.close()
        return True
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Cannot connect to PostgreSQL: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure PostgreSQL is running:")
        print("     Start menu → search 'Services' → find 'postgresql-x64-16' → Start")
        print("  2. Make sure your .env has the correct POSTGRES_PASSWORD")
        print("  3. Make sure port 5432 is not blocked by a firewall")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Hospital Readmission — Database Setup")
    print("=" * 60)

    creds = get_pg_credentials()

    if not creds["password"]:
        print("\nERROR: POSTGRES_PASSWORD is not set in your .env file.")
        print("Open .env and set:  POSTGRES_PASSWORD=<your_postgres_password>")
        sys.exit(1)

    print(f"\nTarget database : {creds['dbname']}")
    print(f"Host            : {creds['host']}:{creds['port']}")
    print(f"User            : {creds['user']}")
    print()

    # Step 1: Test connection
    if not test_connection(creds):
        sys.exit(1)

    print("  Connection: OK")

    # Step 2: Create database
    create_database(creds)

    # Step 3: Apply schema
    apply_schema(creds)

    print("\n" + "=" * 60)
    print("DATABASE SETUP COMPLETE")
    print("=" * 60)
    print("\nNext step — load the pipeline data into PostgreSQL:")
    print("  python src/database/loader.py")
    print()
