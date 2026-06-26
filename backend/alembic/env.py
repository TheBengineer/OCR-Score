"""Async Alembic environment configuration for OCRScore.

Uses SQLAlchemy async engine via asyncpg, compatible with the project's
async-only database layer.
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure the project root is on sys.path so that ``backend`` is importable
# when Alembic runs from the backend/ directory.
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import all models so that Base.metadata is fully populated
import backend.models  # noqa: E402, F401
from backend.database import Base  # noqa: E402
from backend.settings import settings  # noqa: E402

# Alembic Config object
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData for autogenerate support
target_metadata = Base.metadata

# Exclude internal PostgreSQL schemas
exclude_schemas = {"information_schema", "pg_catalog"}


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with the target metadata and emits SQL as a script
    rather than executing it against a live database.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with a live connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations within a connection."""
    connectable = create_async_engine(settings.database_url)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using the async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
