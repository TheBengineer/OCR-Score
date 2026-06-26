"""EngineSecret model — stores API keys and credentials per engine.

Secrets are key-value pairs scoped to an engine slug.  The values are
injected into the engine's ``config`` dict before ``process_pdf`` runs,
so engines find them alongside normal configuration parameters.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class EngineSecret(Base):
    """A single secret (API key, credential) for a specific engine.

    Each record pairs a ``key`` (matching ``SecretDef.key`` from the
    engine's declaration) with its ``value``, scoped to an ``engine_slug``.
    """

    __tablename__ = "engine_secrets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    engine_slug: Mapped[str] = mapped_column(
        String(127),
        nullable=False,
        comment="Engine slug (e.g. 'gcp-document-ai')",
    )
    key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Secret key matching SecretDef.key",
    )
    value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        comment="Secret value (API key, credential path, etc.)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "engine_slug", "key", name="uq_engine_secret_slug_key"
        ),
    )

    def __repr__(self) -> str:
        return f"<EngineSecret slug={self.engine_slug!r} key={self.key!r}>"
