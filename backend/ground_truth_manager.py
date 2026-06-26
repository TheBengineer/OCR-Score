"""Ground truth manager — create, version, and edit ground truth for OCRScore.

Handles both manual and consensus-driven ground truth creation, with full
versioning: each page or word edit creates a new :class:`GroundTruthVersion`
so the audit trail is preserved.

SIZE_OK — Single cohesive manager class (all CRUD + versioning + consensus
integration methods are tightly coupled to the same session and domain).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluation.consensus import build_ground_truth
from backend.models.enums import GroundTruthSource
from backend.models.ground_truth import GroundTruthVersion, GTPageResult
from backend.models.page_result import PageResult


class GroundTruthManagerError(Exception):
    """Raised when a ground truth operation cannot be completed."""


class GroundTruthManager:
    """Business logic for ground truth CRUD and versioning.

    All database access goes through the injected ``AsyncSession`` so the
    manager is testable with an in-memory fake.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Create ──────────────────────────────────────────────────────────────

    async def create_gt_version(
        self,
        pdf_id: uuid.UUID,
        source: GroundTruthSource,
        engine_ids: list[uuid.UUID] | None = None,
        notes: str | None = None,
        created_by: str | None = None,
    ) -> GroundTruthVersion:
        """Create a new ground truth version for a PDF.

        Args:
            pdf_id: The PDF document to create ground truth for.
            source: Provenance of the GT (``manual``, ``consensus``, ``imported``).
            engine_ids: When *source* is ``consensus``, the OCR run UUIDs whose
                page results are used to build consensus GT.
            notes: Optional human-readable notes about this version.
            created_by: Optional identifier of the user or system that created
                this version.

        Returns:
            The newly created :class:`GroundTruthVersion`.

        Raises:
            GroundTruthManagerError: If consensus GT creation fails (e.g., no
                engine outputs found).
        """
        # Determine the next version number for this PDF.
        max_version = await self._max_version_for_pdf(pdf_id)
        version_number = max_version + 1

        gt = GroundTruthVersion(
            pdf_id=pdf_id,
            version_number=version_number,
            source=source,
            notes=notes,
            created_by=created_by,
        )
        self.db.add(gt)
        await self.db.flush()  # Assign gt.id

        # -- consensus path --------------------------------------------------
        if source == GroundTruthSource.CONSENSUS and engine_ids:
            await self._build_consensus_pages(gt, engine_ids)

        await self.db.commit()
        await self.db.refresh(gt)
        return gt

    async def _build_consensus_pages(
        self,
        gt: GroundTruthVersion,
        engine_ids: list[uuid.UUID],
    ) -> None:
        """Run consensus entropy on engine outputs and store GT pages.

        Fetches page results for each run, groups by page number, and
        passes them to :func:`build_ground_truth`.
        """
        # Fetch all page results for the given run IDs.
        result = await self.db.execute(
            select(PageResult).where(PageResult.run_id.in_(engine_ids)),
        )
        page_results: list[PageResult] = list(result.scalars().all())

        if not page_results:
            raise GroundTruthManagerError(
                "No page results found for the given engine run IDs. "
                "Run OCR on the PDF first."
            )

        # Group page results by page number.
        by_page: dict[int, list[dict]] = {}
        for pr in page_results:
            by_page.setdefault(pr.page_number, []).append(pr.data)

        # Build consensus GT per page.
        for page_num, engine_outputs in by_page.items():
            # Pass each engine output as a dict with the canonical schema.
            consensus = build_ground_truth(engine_outputs)
            if consensus["source"] != "auto_consensus":
                # High entropy — no consensus was built.
                continue

            for page_data in consensus["pages"]:
                page_data["consensus_entropy"] = consensus["consensus_entropy"]
                page_data["needs_review"] = consensus["needs_review"]

                gt_page = GTPageResult(
                    gt_version_id=gt.id,
                    page_number=page_num,
                    data=page_data,
                    confidence=1.0 - consensus.get("consensus_entropy", 0.0),
                )
                self.db.add(gt_page)

    # ── Read ────────────────────────────────────────────────────────────────

    async def get_gt_version(self, gt_id: uuid.UUID) -> GroundTruthVersion | None:
        """Fetch a ground truth version with its page results eager-loaded.

        Args:
            gt_id: UUID of the :class:`GroundTruthVersion` to fetch.

        Returns:
            The matching version, or ``None`` if not found.
        """
        result = await self.db.execute(
            select(GroundTruthVersion)
            .where(
                GroundTruthVersion.id == gt_id,
                GroundTruthVersion.deleted_at.is_(None),
            ),
        )
        return result.scalars().one_or_none()

    async def list_gt_versions(
        self,
        pdf_id: uuid.UUID | None = None,
    ) -> list[GroundTruthVersion]:
        """List ground truth versions, optionally filtered by PDF.

        Soft-deleted versions are excluded. Ordered by version number
        descending (newest first).

        Args:
            pdf_id: Optional PDF UUID to filter by.

        Returns:
            List of matching :class:`GroundTruthVersion` records.
        """
        query = select(GroundTruthVersion).where(
            GroundTruthVersion.deleted_at.is_(None),
        )
        if pdf_id is not None:
            query = query.where(GroundTruthVersion.pdf_id == pdf_id)

        result = await self.db.execute(query)
        versions = list(result.scalars().all())
        versions.sort(key=lambda v: v.version_number, reverse=True)
        return versions

    async def get_current_gt(self, pdf_id: uuid.UUID) -> GroundTruthVersion | None:
        """Return the current (most recently promoted) ground truth for a PDF.

        "Current" is defined as the non-deleted version with the highest
        ``version_number`` for the given PDF.

        Args:
            pdf_id: The PDF UUID.

        Returns:
            The current :class:`GroundTruthVersion`, or ``None``.
        """
        result = await self.db.execute(
            select(GroundTruthVersion).where(
                GroundTruthVersion.pdf_id == pdf_id,
                GroundTruthVersion.deleted_at.is_(None),
            ),
        )
        versions = result.scalars().all()
        if not versions:
            return None
        return max(versions, key=lambda v: v.version_number)

    # ── Update (versioned) ──────────────────────────────────────────────────

    async def update_gt_page(
        self,
        gt_id: uuid.UUID,
        page_num: int,
        page_data: dict,
    ) -> GTPageResult:
        """Update a single page's ground truth data, creating a new version.

        The existing version is preserved for audit. A new
        :class:`GroundTruthVersion` is created with incremented version number,
        copying unchanged pages from the source version.

        Args:
            gt_id: The source GT version ID to branch from.
            page_num: The page number to update (1-based).
            page_data: New canonical JSONB page data.

        Returns:
            The :class:`GTPageResult` from the *new* version.

        Raises:
            GroundTruthManagerError: If the source version is not found,
                is soft-deleted, or the page does not exist in the source.
        """
        source = await self._get_active_gt(gt_id)
        return await self._create_versioned_edit(
            source=source,
            updated_page_num=page_num,
            updated_page_data=page_data,
            change_summary=f"Updated page {page_num}",
        )

    async def update_gt_word(
        self,
        gt_id: uuid.UUID,
        page_num: int,
        word_idx: int,
        new_text: str,
    ) -> GTPageResult:
        """Correct a single word in ground truth, creating a new version.

        Navigates into the canonical JSONB hierarchy (blocks → lines → words)
        to find the word at *word_idx* within the first text block on the page,
        then replaces its text.

        Args:
            gt_id: The source GT version ID to branch from.
            page_num: The page number containing the word (1-based).
            word_idx: 0-based index of the word within the first text block's
                first line.
            new_text: The corrected text for the word.

        Returns:
            The :class:`GTPageResult` from the *new* version.

        Raises:
            GroundTruthManagerError: If the source version is not found,
                is soft-deleted, or the word index is out of range.
        """
        source = await self._get_active_gt(gt_id)

        # Find the source page result.
        source_page = await self._get_page_result(gt_id, page_num)
        if source_page is None:
            raise GroundTruthManagerError(
                f"Page {page_num} not found in ground truth version {gt_id}"
            )

        # Navigate JSONB to find the word.
        import copy

        new_data = copy.deepcopy(source_page.data)
        blocks: list[dict] = new_data.get("blocks", [])
        if not blocks:
            raise GroundTruthManagerError(
                f"No blocks in page {page_num} data — cannot locate word index"
            )
        lines: list[dict] = blocks[0].get("lines", [])
        if not lines:
            raise GroundTruthManagerError(
                f"No lines in first block of page {page_num} — cannot locate word index"
            )
        words: list[dict] = lines[0].get("words", [])
        if word_idx < 0 or word_idx >= len(words):
            raise GroundTruthManagerError(
                f"Word index {word_idx} out of range (0–{len(words) - 1}) "
                f"on page {page_num}"
            )

        old_text = words[word_idx]["text"]
        words[word_idx]["text"] = new_text

        # Update the line text too for consistency.
        lines[0]["text"] = " ".join(w["text"] for w in words)

        return await self._create_versioned_edit(
            source=source,
            updated_page_num=page_num,
            updated_page_data=new_data,
            change_summary=(
                f"Corrected word[{word_idx}] on page {page_num}: "
                f"'{old_text}' → '{new_text}'"
            ),
        )

    async def _create_versioned_edit(
        self,
        source: GroundTruthVersion,
        updated_page_num: int,
        updated_page_data: dict,
        change_summary: str,
    ) -> GTPageResult:
        """Create a new version by copying ``source`` and replacing one page.

        Args:
            source: The source GT version to branch from.
            updated_page_num: The page number that changed.
            updated_page_data: The new JSONB data for the changed page.
            change_summary: Human-readable description of the change (stored
                in the new version's ``notes`` field, appended).

        Returns:
            The :class:`GTPageResult` for the updated page in the new version.
        """
        new_version_number = (await self._max_version_for_pdf(source.pdf_id)) + 1

        new_gt = GroundTruthVersion(
            pdf_id=source.pdf_id,
            version_number=new_version_number,
            source=source.source,
            created_by=source.created_by,
            notes=(
                f"[v{source.version_number}→v{new_version_number}] {change_summary}"
                + (f"\nPrevious notes: {source.notes}" if source.notes else "")
            ),
        )
        self.db.add(new_gt)
        await self.db.flush()  # Assign new_gt.id

        # Copy unchanged pages, replace the updated one.
        source_pages = await self._get_all_pages(source.id)
        new_page_result: GTPageResult | None = None

        for sp in source_pages:
            if sp.page_number == updated_page_num:
                new_pr = GTPageResult(
                    gt_version_id=new_gt.id,
                    page_number=updated_page_num,
                    width=sp.width,
                    height=sp.height,
                    data=updated_page_data,
                    confidence=sp.confidence,
                )
                self.db.add(new_pr)
                new_page_result = new_pr
            else:
                # Copy unchanged page.
                new_pr = GTPageResult(
                    gt_version_id=new_gt.id,
                    page_number=sp.page_number,
                    width=sp.width,
                    height=sp.height,
                    data=sp.data,
                    confidence=sp.confidence,
                )
                self.db.add(new_pr)

        await self.db.commit()
        await self.db.refresh(new_gt)

        if new_page_result is None:
            raise GroundTruthManagerError(
                f"Page {updated_page_num} not found in source version {source.id}"
            )

        return new_page_result

    # ── Delete ──────────────────────────────────────────────────────────────

    async def soft_delete_gt(self, gt_id: uuid.UUID) -> None:
        """Soft-delete a ground truth version by setting ``deleted_at``.

        Args:
            gt_id: UUID of the :class:`GroundTruthVersion` to delete.

        Raises:
            GroundTruthManagerError: If the version is not found or is
                already deleted.
        """
        gt = await self._get_active_gt(gt_id)
        gt.deleted_at = datetime.now(UTC)
        await self.db.commit()

    # ── Promote ─────────────────────────────────────────────────────────────

    async def promote_gt_version(self, gt_id: uuid.UUID) -> None:
        """Promote a ground truth version to be the "current" one for its PDF.

        "Current" is determined by having the highest ``version_number`` among
        non-deleted versions for the same PDF. This method bumps the version
        number to make this version the current one.

        Args:
            gt_id: UUID of the :class:`GroundTruthVersion` to promote.

        Raises:
            GroundTruthManagerError: If the version is not found, is
                soft-deleted, or is already current.
        """
        gt = await self._get_active_gt(gt_id)
        max_version = await self._max_version_for_pdf(gt.pdf_id)

        if gt.version_number == max_version:
            raise GroundTruthManagerError(
                f"Version {gt_id} (v{gt.version_number}) is already the "
                f"current version for PDF {gt.pdf_id}"
            )

        # Ensure uniqueness: temporarily set to a safe high value, commit,
        # then other versions can keep their numbers.
        new_num = max_version + 1
        gt.version_number = new_num
        gt.notes = (gt.notes or "") + f"\nPromoted to current (v{new_num})"
        await self.db.commit()

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _max_version_for_pdf(self, pdf_id: uuid.UUID) -> int:
        """Return the highest version_number for any GT of this PDF."""
        result = await self.db.execute(
            select(GroundTruthVersion).where(
                GroundTruthVersion.pdf_id == pdf_id,
            ),
        )
        versions = result.scalars().all()
        if not versions:
            return 0
        return max(v.version_number for v in versions)

    async def _get_active_gt(self, gt_id: uuid.UUID) -> GroundTruthVersion:
        """Fetch a non-deleted GT version or raise."""
        result = await self.db.execute(
            select(GroundTruthVersion).where(
                GroundTruthVersion.id == gt_id,
                GroundTruthVersion.deleted_at.is_(None),
            ),
        )
        gt = result.scalars().one_or_none()
        if gt is None:
            # Check if it exists but is deleted to give a better error.
            check = await self.db.execute(
                select(GroundTruthVersion.deleted_at).where(
                    GroundTruthVersion.id == gt_id,
                ),
            )
            existing = check.scalars().one_or_none()
            if existing is not None:
                raise GroundTruthManagerError(
                    f"Ground truth version {gt_id} has been soft-deleted"
                )
            raise GroundTruthManagerError(
                f"Ground truth version {gt_id} not found"
            )
        return gt

    async def _get_page_result(
        self,
        gt_id: uuid.UUID,
        page_num: int,
    ) -> GTPageResult | None:
        """Get a single page result for a GT version."""
        result = await self.db.execute(
            select(GTPageResult).where(
                GTPageResult.gt_version_id == gt_id,
                GTPageResult.page_number == page_num,
            ),
        )
        return result.scalars().one_or_none()

    async def _get_all_pages(self, gt_id: uuid.UUID) -> list[GTPageResult]:
        """Get all page results for a GT version, ordered by page number."""
        result = await self.db.execute(
            select(GTPageResult).where(
                GTPageResult.gt_version_id == gt_id,
            ),
        )
        pages = list(result.scalars().all())
        pages.sort(key=lambda p: p.page_number)
        return pages
