# ruff: noqa: E501
"""Initial migration — create all OCRScore tables.

Revision ID: 001
Revises: None
Create Date: 2026-06-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── Create ENUM types ───────────────────────────────────────────────────
    sa.Enum("uploading", "uploaded", "processing", "ready", "error", "deleted", name="pdf_status").create(op.get_bind())
    sa.Enum("pending", "queued", "running", "completed", "failed", "cancelled", name="run_status").create(op.get_bind())
    sa.Enum("manual", "consensus", "imported", name="ground_truth_source").create(op.get_bind())
    sa.Enum(
        "character", "word", "line", "paragraph", "block", "table", "page", "document",
        name="score_level",
    ).create(op.get_bind())
    sa.Enum(
        "cer", "wer", "accuracy", "precision", "recall", "f1",
        "edit_distance", "confidence", "table_structure",
        name="score_metric",
    ).create(op.get_bind())

    # ── Table: pdfs ─────────────────────────────────────────────────────────
    op.create_table(
        "pdfs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(1024), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("md5_hash", sa.String(32), nullable=False, index=True),
        sa.Column("sha256_hash", sa.String(64), nullable=False, index=True),
        sa.Column("mime_type", sa.String(127), nullable=False, server_default=sa.text("'application/pdf'")),
        sa.Column(
            "status",
            postgresql.ENUM("uploading", "uploaded", "processing", "ready", "error", "deleted", name="pdf_status", create_type=False),
            nullable=False,
            server_default=sa.text("'uploading'"),
        ),
        sa.Column("upload_timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_pdfs_status", "pdfs", ["status"])
    op.create_index("ix_pdfs_upload_timestamp", "pdfs", ["upload_timestamp"])
    op.create_index(
        "ix_pdfs_md5_sha256",
        "pdfs",
        ["md5_hash", "sha256_hash"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # ── Table: ocr_engines ──────────────────────────────────────────────────
    op.create_table(
        "ocr_engines",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("slug", sa.String(127), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(63), nullable=False, server_default=sa.text("'0.0.0'")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("config_schema", postgresql.JSONB(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # ── Table: ground_truth_versions ────────────────────────────────────────
    op.create_table(
        "ground_truth_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("pdf_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("pdfs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "source",
            postgresql.ENUM("manual", "consensus", "imported", name="ground_truth_source", create_type=False),
            nullable=False,
            server_default=sa.text("'manual'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_gt_versions_pdf_id", "ground_truth_versions", ["pdf_id"])
    op.create_index(
        "ix_gt_versions_pdf_version",
        "ground_truth_versions",
        ["pdf_id", "version_number"],
        unique=True,
    )

    # ── Table: ocr_runs ─────────────────────────────────────────────────────
    op.create_table(
        "ocr_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("pdf_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("pdfs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("engine_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ocr_engines.id", ondelete="RESTRICT"), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("pending", "queued", "running", "completed", "failed", "cancelled", name="run_status", create_type=False),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("engine_config", postgresql.JSONB(), nullable=True),
        sa.Column("engine_version", sa.String(63), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("raw_output_uri", sa.String(2048), nullable=True),
        sa.Column("run_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("environment", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_index("ix_ocr_runs_pdf_id", "ocr_runs", ["pdf_id"])
    op.create_index("ix_ocr_runs_engine_id", "ocr_runs", ["engine_id"])
    op.create_index("ix_ocr_runs_status", "ocr_runs", ["status"])
    op.create_index("ix_ocr_runs_created_at", "ocr_runs", ["created_at"])

    # ── Table: page_results ─────────────────────────────────────────────────
    op.create_table(
        "page_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ocr_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=True),
        sa.Column("height", sa.Float(), nullable=True),
        sa.Column("data", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("confidence", sa.Float(), nullable=True),
    )

    op.create_index("ix_page_results_run_id", "page_results", ["run_id"])
    op.create_index(
        "ix_page_results_run_page",
        "page_results",
        ["run_id", "page_number"],
        unique=True,
    )
    op.create_index(
        "ix_page_results_data_gin",
        "page_results",
        ["data"],
        postgresql_using="gin",
        postgresql_ops={"data": "jsonb_path_ops"},
    )

    # ── Table: gt_page_results ──────────────────────────────────────────────
    op.create_table(
        "gt_page_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "gt_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ground_truth_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=True),
        sa.Column("height", sa.Float(), nullable=True),
        sa.Column("data", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("confidence", sa.Float(), nullable=True, server_default=sa.text("1.0")),
    )

    op.create_index("ix_gt_page_results_gt_version_id", "gt_page_results", ["gt_version_id"])
    op.create_index(
        "ix_gt_page_results_version_page",
        "gt_page_results",
        ["gt_version_id", "page_number"],
        unique=True,
    )
    op.create_index(
        "ix_gt_page_results_data_gin",
        "gt_page_results",
        ["data"],
        postgresql_using="gin",
        postgresql_ops={"data": "jsonb_path_ops"},
    )

    # ── Table: scores ───────────────────────────────────────────────────────
    op.create_table(
        "scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ocr_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "gt_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ground_truth_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "level",
            postgresql.ENUM("character", "word", "line", "paragraph", "block", "table", "page", "document", name="score_level", create_type=False),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column(
            "metric",
            postgresql.ENUM("cer", "wer", "accuracy", "precision", "recall", "f1", "edit_distance", "confidence", "table_structure", name="score_metric", create_type=False),
            nullable=False,
        ),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("confidence_weighted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_index("ix_scores_run_id", "scores", ["run_id"])
    op.create_index("ix_scores_gt_version_id", "scores", ["gt_version_id"])
    op.create_index("ix_scores_level_metric", "scores", ["level", "metric"])

    # ── Table: score_summaries ──────────────────────────────────────────────
    op.create_table(
        "score_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ocr_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "gt_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ground_truth_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("breakdown", postgresql.JSONB(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_index("ix_score_summaries_run_id", "score_summaries", ["run_id"])
    op.create_index("ix_score_summaries_gt_version_id", "score_summaries", ["gt_version_id"])


def downgrade() -> None:
    """Drop all tables and enum types in reverse order."""
    op.drop_table("score_summaries")
    op.drop_table("scores")
    op.drop_table("gt_page_results")
    op.drop_table("page_results")
    op.drop_table("ocr_runs")
    op.drop_table("ground_truth_versions")
    op.drop_table("ocr_engines")
    op.drop_table("pdfs")

    op.execute("DROP TYPE IF EXISTS score_metric")
    op.execute("DROP TYPE IF EXISTS score_level")
    op.execute("DROP TYPE IF EXISTS ground_truth_source")
    op.execute("DROP TYPE IF EXISTS run_status")
    op.execute("DROP TYPE IF EXISTS pdf_status")
