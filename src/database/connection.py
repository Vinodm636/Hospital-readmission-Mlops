"""
connection.py
-------------
SQLAlchemy database connection module for the Hospital Readmission project.

What this module does:
    Provides a single, reusable SQLAlchemy engine and session factory.
    All database credentials are loaded from environment variables (.env)
    — never hardcoded.

Why we need it:
    Every module that writes to or reads from PostgreSQL should use the
    same connection factory.  This avoids duplicated connection code and
    ensures credentials are managed in one place.

Viva questions:
    Q: What is SQLAlchemy and why use it instead of raw psycopg2?
    A: SQLAlchemy is an ORM (Object-Relational Mapper) that lets us write
       Python code to interact with the database instead of raw SQL strings.
       It also handles connection pooling and is database-agnostic.

    Q: What is connection pooling?
    A: Reusing a fixed number of database connections rather than opening
       and closing one per query. This is faster and uses fewer resources.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ── Project root path setup ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from src.utils.logger import get_logger

load_dotenv(_PROJECT_ROOT / ".env")

logger = get_logger(__name__)


def build_connection_url(
    host: Optional[str] = None,
    port: Optional[str] = None,
    db: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """
    Build a PostgreSQL connection URL from environment variables.

    Reads from .env / environment. Raises if required vars are missing.

    Returns:
        str: PostgreSQL connection string in SQLAlchemy format.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    host = host or os.getenv("POSTGRES_HOST", "localhost")
    port = port or os.getenv("POSTGRES_PORT", "5432")
    db = db or os.getenv("POSTGRES_DB")
    user = user or os.getenv("POSTGRES_USER")
    password = password or os.getenv("POSTGRES_PASSWORD")

    missing = [
        name for name, val in
        [("POSTGRES_DB", db), ("POSTGRES_USER", user), ("POSTGRES_PASSWORD", password)]
        if not val
    ]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {missing}\n"
            "Create a .env file from .env.example and fill in your credentials."
        )

    # postgresql+psycopg2://user:password@host:port/database
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return url


@lru_cache(maxsize=1)
def get_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """
    Create and return a cached SQLAlchemy engine.

    The @lru_cache decorator ensures only ONE engine is created per
    process — subsequent calls return the cached instance.

    Args:
        pool_size (int): Number of persistent connections in the pool.
        max_overflow (int): Extra connections allowed beyond pool_size.

    Returns:
        Engine: SQLAlchemy Engine instance.
    """
    url = build_connection_url()
    # Log a sanitised version (without password)
    safe_url = url.split("@")[1] if "@" in url else url
    logger.info("Creating database engine: ...@%s", safe_url)

    engine = create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,       # test connection before using from pool
        echo=False,               # set True to log all SQL (verbose)
    )
    return engine


def get_session_factory() -> sessionmaker:
    """
    Return a SQLAlchemy session factory bound to the shared engine.

    Returns:
        sessionmaker: Session factory for creating database sessions.
    """
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def test_connection() -> bool:
    """
    Test that the database connection is working by running a trivial query.

    Returns:
        bool: True if connection succeeds, False otherwise.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                logger.info("Database connection test: PASSED")
                return True
    except Exception as exc:
        logger.error("Database connection test: FAILED — %s", exc)
        return False
    return False


def get_schema_names() -> list:
    """
    Return the list of schemas that should exist after schema.sql is applied.

    Returns:
        list: Expected schema names.
    """
    return ["raw", "staging", "curated", "analytics"]


if __name__ == "__main__":
    # Quick connectivity test
    print("Testing PostgreSQL connection...")
    ok = test_connection()
    print(f"Connection: {'OK' if ok else 'FAILED'}")
