"""SQLAlchemy async engine, session factory, and declarative base for OCRScore."""

from collections.abc import AsyncGenerator

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.settings import settings

__all__: list[str] = [
    "Base",
    "engine",
    "async_session_factory",
    "get_db_session",
]

engine = create_async_engine(
    settings.database_url,
    poolclass=NullPool,
    echo=settings.echo_sql,
    connect_args={
        "statement_cache_size": 0,  # asyncpg default
    },
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all OCRScore models."""
    pass


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency yielding an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise
        finally:
            await session.close()
