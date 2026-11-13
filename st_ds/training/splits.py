"""Leakage-resistant grouped dataset splitting (Section 19 & 31, DEF-009).

Forbids row-level random train/test splits. Ensures that replicates of the same
soil sample (`sample_id`), paired `wet/dry` scans, and samples from the same
parcel/location (`site_id`) never cross between training and validation/test sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from st_ds.domain.errors import SchemaError
from st_ds.domain.schemas import Measurement
from st_ds.io.dataset import SoilDataset

GroupKeyType = Literal["site_id", "sample_id", "session_id"]


@dataclass(frozen=True)
class SplitFold:
    """Single leakage-verified train/validation split fold."""

    fold_index: int
    group_key: GroupKeyType
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    val_sample_ids: tuple[str, ...]


class GroupedSplitBuilder:
    """Constructs and verifies leakage-free GroupKFold and Leave-One-Site-Out splits."""

    @staticmethod
    def verify_no_leakage(
        train_records: Sequence[Measurement],
        eval_records: Sequence[Measurement],
        *,
        enforce_site_disjoint: bool = True,
    ) -> None:
        """Mathematically verify zero sample_id and (optionally) zero site_id overlap."""
        if len(train_records) == 0 or len(eval_records) == 0:
            raise SchemaError("Split contains an empty train or evaluation partition.")

        train_samples = {r.sample_id for r in train_records}
        eval_samples = {r.sample_id for r in eval_records}
        sample_overlap = sorted(train_samples & eval_samples)
        if sample_overlap:
            raise SchemaError(
                f"Data leakage detected! {len(sample_overlap)} sample_id(s) appear in both train and eval sets.",
                details={"overlapping_sample_ids": sample_overlap[:10]},
            )

        if enforce_site_disjoint:
            train_sites = {r.site_id for r in train_records}
            eval_sites = {r.site_id for r in eval_records}
            site_overlap = sorted(train_sites & eval_sites)
            if site_overlap:
                raise SchemaError(
                    f"Site leakage detected! {len(site_overlap)} site_id(s) appear in both train and eval sets.",
                    details={"overlapping_site_ids": site_overlap},
                )

    @classmethod
    def group_kfold_split(
        cls,
        dataset: SoilDataset,
        *,
        n_splits: int = 5,
        group_by: GroupKeyType = "site_id",
    ) -> list[SplitFold]:
        """Split a SoilDataset into `n_splits` group-disjoint folds."""
        if n_splits < 2:
            raise SchemaError(f"n_splits must be >= 2, got {n_splits}.")

        groups = np.asarray([getattr(r, group_by) for r in dataset.records], dtype=str)
        unique_groups = np.unique(groups)
        if unique_groups.size < n_splits:
            raise SchemaError(
                f"Cannot perform {n_splits}-fold GroupKFold on '{group_by}' with only "
                f"{unique_groups.size} unique groups: {list(unique_groups)}.",
                details={"n_splits": n_splits, "n_unique_groups": int(unique_groups.size)},
            )

        # Also check that no single sample_id spans across multiple groups
        sample_to_groups: dict[str, set[str]] = {}
        for r, g in zip(dataset.records, groups):
            sample_to_groups.setdefault(r.sample_id, set()).add(str(g))
        inconsistent = {s: sorted(gs) for s, gs in sample_to_groups.items() if len(gs) > 1}
        if inconsistent:
            raise SchemaError(
                f"Inconsistent group assignment: sample_id(s) map to multiple '{group_by}' values.",
                details={"inconsistent_samples": inconsistent},
            )

        gkf = GroupKFold(n_splits=n_splits)
        dummy_x = np.zeros(len(dataset.records))
        folds: list[SplitFold] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(dummy_x, dummy_x, groups=groups)):
            tr_recs = [dataset.records[int(i)] for i in tr_idx]
            val_recs = [dataset.records[int(i)] for i in val_idx]
            cls.verify_no_leakage(
                tr_recs,
                val_recs,
                enforce_site_disjoint=(group_by == "site_id"),
            )
            folds.append(
                SplitFold(
                    fold_index=fold_idx,
                    group_key=group_by,
                    train_indices=tuple(int(i) for i in tr_idx),
                    val_indices=tuple(int(i) for i in val_idx),
                    train_groups=tuple(sorted({str(groups[i]) for i in tr_idx})),
                    val_groups=tuple(sorted({str(groups[i]) for i in val_idx})),
                    train_sample_ids=tuple(sorted({r.sample_id for r in tr_recs})),
                    val_sample_ids=tuple(sorted({r.sample_id for r in val_recs})),
                )
            )
        return folds

    @classmethod
    def leave_one_site_out_split(cls, dataset: SoilDataset) -> list[SplitFold]:
        """Leave-One-Site-Out (LOSO) spatial/site generalization split."""
        groups = np.asarray([r.site_id for r in dataset.records], dtype=str)
        unique_sites = np.unique(groups)
        if unique_sites.size < 2:
            raise SchemaError(
                f"Leave-One-Site-Out requires at least 2 distinct site_ids, found {unique_sites.size}."
            )

        logo = LeaveOneGroupOut()
        dummy_x = np.zeros(len(dataset.records))
        folds: list[SplitFold] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(logo.split(dummy_x, dummy_x, groups=groups)):
            tr_recs = [dataset.records[int(i)] for i in tr_idx]
            val_recs = [dataset.records[int(i)] for i in val_idx]
            cls.verify_no_leakage(tr_recs, val_recs, enforce_site_disjoint=True)
            folds.append(
                SplitFold(
                    fold_index=fold_idx,
                    group_key="site_id",
                    train_indices=tuple(int(i) for i in tr_idx),
                    val_indices=tuple(int(i) for i in val_idx),
                    train_groups=tuple(sorted({str(groups[i]) for i in tr_idx})),
                    val_groups=tuple(sorted({str(groups[i]) for i in val_idx})),
                    train_sample_ids=tuple(sorted({r.sample_id for r in tr_recs})),
                    val_sample_ids=tuple(sorted({r.sample_id for r in val_recs})),
                )
            )
        return folds
