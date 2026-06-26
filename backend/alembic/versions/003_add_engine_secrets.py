# ruff: noqa: E501
"""Add engine_secrets table for storing API keys and credentials.

Revision ID: 003
Revises: 002
Create Date: 2026-06-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "engine_secrets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("engine_slug", sa.String(127), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("value", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("engine_slug", "key", name="uq_engine_secret_slug_key"),
    )


def downgrade() -> None:
    op.drop_table("engine_secrets")
