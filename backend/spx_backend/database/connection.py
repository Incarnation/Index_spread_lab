from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from spx_backend.config import settings


def create_engine() -> AsyncEngine:
    """Create shared async SQLAlchemy engine."""
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


engine = create_engine()
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncSession:
    """FastAPI dependency that yields one async DB session."""
    async with SessionLocal() as session:
        yield session

