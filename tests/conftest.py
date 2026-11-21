import os
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

from rag.__main__ import LOCAL_DSN


@pytest.fixture
def db():
    """Each test gets its own PostgreSQL schema and rolls back its changes."""
    with psycopg.connect(os.environ.get("RAG_DATABASE_URL", LOCAL_DSN)) as conn:
        with conn.transaction(force_rollback=True):
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            schema = sql.Identifier("rag_test_" + uuid4().hex)
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
            conn.execute(sql.SQL("SET LOCAL search_path TO {}, public").format(schema))
            yield conn
