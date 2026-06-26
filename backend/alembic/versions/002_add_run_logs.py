# ruff: noqa: E501
"""Add logs JSONB column to ocr_runs table.

Revision ID: 002
Revises: 001
Create Date: 2026-06-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ocr_runs",
        sa.Column(
            "logs",
            postgresql.JSONB,
            nullable=True,
            comment="Structured log entries: [{timestamp, level, message}, ...]",
        ),
    )


def downgrade() -> None:
    op.drop_column("ocr_runs", "logs")
