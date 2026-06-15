"""Dataset Exporter — Phase 3.1.

Exports the compiled Atarashi and Nirjas datasets to Parquet-backed
Hugging Face ``datasets.Dataset`` format.  Each exported dataset is a
directory containing:

* ``data-00000-of-00001.parquet`` — the data shard(s)
* ``dataset_info.json`` — HF metadata (features, description, …)
* ``state.json`` — split info

The datasets can be loaded back with:

    >>> from datasets import load_from_disk
    >>> ds = load_from_disk("output/atarashi")

Exports
-------
* **Atarashi dataset** — licence texts with labels for similarity-based
  training.  Columns: ``license_key``, ``text``, ``source``.
* **Nirjas dataset** — comment texts with 2-class labels.  Columns:
  ``text``, ``label``, ``source``, ``negative_type``.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from pydantic import BaseModel, Field

try:
    from ..builder.augmented_merge import AtarashiSample, NirjasSample
except ImportError:  # pragma: no cover — direct-script execution
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from builder.augmented_merge import AtarashiSample, NirjasSample

logger = logging.getLogger(__name__)


# Configuration


class ExportConfig(BaseModel):
    """Configuration knobs for the dataset exporter."""

    model_config = {"protected_namespaces": ()}

    output_dir: str = Field(
        default="output",
        description="Root directory for all exported datasets.",
    )
    atarashi_dir_name: str = Field(
        default="atarashi",
        description="Sub-directory name for the Atarashi dataset.",
    )
    nirjas_dir_name: str = Field(
        default="nirjas",
        description="Sub-directory name for the Nirjas dataset.",
    )
    train_split_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of samples assigned to the 'train' split.  "
            "The remainder goes to 'test'.  Set to 1.0 to export a "
            "single 'train' split only."
        ),
    )
    random_seed: int = 42
    write_statistics: bool = Field(
        default=True,
        description="Write a ``statistics.json`` alongside each exported dataset.",
    )


# Export result


class ExportResult(BaseModel):
    """Summary returned after an export run."""

    model_config = {"protected_namespaces": ()}

    atarashi_path: Optional[str] = None
    nirjas_path: Optional[str] = None
    atarashi_total: int = 0
    atarashi_train: int = 0
    atarashi_test: int = 0
    nirjas_total: int = 0
    nirjas_train: int = 0
    nirjas_test: int = 0


# HF Features schemas

ATARASHI_FEATURES = Features(
    {
        "license_key": Value("string"),
        "text": Value("string"),
        "source": Value("string"),
    }
)

NIRJAS_FEATURES = Features(
    {
        "text": Value("string"),
        "label": ClassLabel(names=["license_related", "not_license_related"]),
        "source": Value("string"),
        "negative_type": Value("string"),
    }
)


# Core exporter


class DatasetExporter:
    """Export Atarashi and Nirjas datasets as HF ``Dataset`` on disk.

    Parameters
    ----------
    config : ExportConfig, optional
        Export configuration.  Uses defaults if omitted.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    # -- Atarashi -----------------------------------------------------------

    @staticmethod
    def _atarashi_to_dict(samples: list[AtarashiSample]) -> dict[str, list]:
        """Convert ``AtarashiSample`` list into a column-oriented dict."""
        return {
            "license_key": [s.license_key for s in samples],
            "text": [s.text for s in samples],
            "source": [s.source for s in samples],
        }

    def _build_atarashi_dataset(
        self,
        samples: list[AtarashiSample],
    ) -> DatasetDict | Dataset:
        """Build an HF Atarashi DatasetDict with per-license splitting.

        A global random split can place all samples for a rare ``license_key`` in
        the test set.  For multi-class license identification that creates
        impossible evaluation labels.  Split each license independently and keep
        singleton classes in train only, ensuring every test label is seen in
        train.
        """
        if self.config.train_split_ratio >= 1.0:
            ds = Dataset.from_dict(
                self._atarashi_to_dict(samples),
                features=ATARASHI_FEATURES,
            )
            return DatasetDict({"train": ds})

        rng = random.Random(self.config.random_seed)
        by_key: dict[str, list[AtarashiSample]] = defaultdict(list)
        for sample in samples:
            by_key[sample.license_key].append(sample)

        train: list[AtarashiSample] = []
        test: list[AtarashiSample] = []
        test_ratio = 1.0 - self.config.train_split_ratio

        for group in by_key.values():
            group = list(group)
            rng.shuffle(group)
            if len(group) == 1 or test_ratio <= 0.0:
                train.extend(group)
                continue

            n_test = max(1, round(len(group) * test_ratio))
            n_test = min(n_test, len(group) - 1)  # leave at least one in train
            test.extend(group[:n_test])
            train.extend(group[n_test:])

        rng.shuffle(train)
        rng.shuffle(test)

        train_ds = Dataset.from_dict(
            self._atarashi_to_dict(train),
            features=ATARASHI_FEATURES,
        )
        test_ds = Dataset.from_dict(
            self._atarashi_to_dict(test),
            features=ATARASHI_FEATURES,
        )
        return DatasetDict({"train": train_ds, "test": test_ds})

    # -- Nirjas -------------------------------------------------------------

    @staticmethod
    def _nirjas_to_dict(samples: list[NirjasSample]) -> dict[str, list]:
        """Convert ``NirjasSample`` list into a column-oriented dict."""
        return {
            "text": [s.text for s in samples],
            "label": [s.label for s in samples],
            "source": [s.source for s in samples],
            "negative_type": [s.negative_type or "" for s in samples],
        }

    def _build_nirjas_dataset(
        self,
        samples: list[NirjasSample],
    ) -> DatasetDict | Dataset:
        """Build an HF ``DatasetDict`` (train/test) or single ``Dataset``."""
        ds = Dataset.from_dict(
            self._nirjas_to_dict(samples),
            features=NIRJAS_FEATURES,
        )
        if self.config.train_split_ratio >= 1.0:
            return DatasetDict({"train": ds})

        split = ds.train_test_split(
            test_size=1.0 - self.config.train_split_ratio,
            seed=self.config.random_seed,
            stratify_by_column="label",
        )
        return split

    # -- Statistics ---------------------------------------------------------

    @staticmethod
    def _atarashi_statistics(ds: DatasetDict) -> dict:
        """Compute summary statistics for the Atarashi export."""
        stats: dict = {"splits": {}}
        total = 0
        all_keys: set[str] = set()
        source_counts: dict[str, int] = {}

        for split_name, split_ds in ds.items():
            n = len(split_ds)
            total += n
            keys = set(split_ds["license_key"])
            all_keys.update(keys)
            split_sources: dict[str, int] = {}
            for src in split_ds["source"]:
                split_sources[src] = split_sources.get(src, 0) + 1
                source_counts[src] = source_counts.get(src, 0) + 1
            stats["splits"][split_name] = {
                "samples": n,
                "unique_licenses": len(keys),
                "by_source": split_sources,
            }

        stats["total_samples"] = total
        stats["total_unique_licenses"] = len(all_keys)
        stats["by_source"] = source_counts
        return stats

    @staticmethod
    def _nirjas_statistics(ds: DatasetDict) -> dict:
        """Compute summary statistics for the Nirjas export."""
        stats: dict = {"splits": {}}
        total = 0
        label_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        neg_type_counts: dict[str, int] = {}

        label_names = ds[list(ds.keys())[0]].features["label"].names

        for split_name, split_ds in ds.items():
            n = len(split_ds)
            total += n
            split_labels: dict[str, int] = {}
            split_sources: dict[str, int] = {}
            split_neg: dict[str, int] = {}
            for i in range(n):
                row = split_ds[i]
                label_str = (
                    label_names[row["label"]]
                    if isinstance(row["label"], int)
                    else row["label"]
                )
                split_labels[label_str] = split_labels.get(label_str, 0) + 1
                label_counts[label_str] = label_counts.get(label_str, 0) + 1
                split_sources[row["source"]] = split_sources.get(row["source"], 0) + 1
                source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
                neg = row.get("negative_type", "")
                if neg:
                    split_neg[neg] = split_neg.get(neg, 0) + 1
                    neg_type_counts[neg] = neg_type_counts.get(neg, 0) + 1
            stats["splits"][split_name] = {
                "samples": n,
                "by_label": split_labels,
                "by_source": split_sources,
                "by_negative_type": split_neg,
            }

        stats["total_samples"] = total
        stats["by_label"] = label_counts
        stats["by_source"] = source_counts
        stats["by_negative_type"] = neg_type_counts
        return stats

    # -- Main export --------------------------------------------------------

    def export(
        self,
        atarashi_samples: Optional[list[AtarashiSample]] = None,
        nirjas_samples: Optional[list[NirjasSample]] = None,
    ) -> ExportResult:
        """Export one or both datasets to disk.

        Parameters
        ----------
        atarashi_samples :
            Atarashi licence-similarity samples from ``AugmentedMerger``.
        nirjas_samples :
            Nirjas 2-class comment-classification samples.

        Returns
        -------
        ExportResult
            Paths and counts for each exported dataset.
        """
        root = Path(self.config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        result = ExportResult()

        # --- Atarashi -------------------------------------------------------
        if atarashi_samples:
            atarashi_path = root / self.config.atarashi_dir_name
            ds_dict = self._build_atarashi_dataset(atarashi_samples)
            if isinstance(ds_dict, Dataset):
                ds_dict = DatasetDict({"train": ds_dict})
            ds_dict.save_to_disk(str(atarashi_path))
            result.atarashi_path = str(atarashi_path)
            result.atarashi_total = len(atarashi_samples)
            result.atarashi_train = len(ds_dict.get("train", []))
            result.atarashi_test = len(ds_dict.get("test", []))

            if self.config.write_statistics:
                stats = self._atarashi_statistics(ds_dict)
                stats_path = atarashi_path / "statistics.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                logger.info("Atarashi statistics written to %s", stats_path)

            logger.info(
                "Atarashi dataset exported: %d total (%d train / %d test) → %s",
                result.atarashi_total,
                result.atarashi_train,
                result.atarashi_test,
                atarashi_path,
            )

        # --- Nirjas ---------------------------------------------------------
        if nirjas_samples:
            nirjas_path = root / self.config.nirjas_dir_name
            ds_dict = self._build_nirjas_dataset(nirjas_samples)
            if isinstance(ds_dict, Dataset):
                ds_dict = DatasetDict({"train": ds_dict})
            ds_dict.save_to_disk(str(nirjas_path))
            result.nirjas_path = str(nirjas_path)
            result.nirjas_total = len(nirjas_samples)
            result.nirjas_train = len(ds_dict.get("train", []))
            result.nirjas_test = len(ds_dict.get("test", []))

            if self.config.write_statistics:
                stats = self._nirjas_statistics(ds_dict)
                stats_path = nirjas_path / "statistics.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                logger.info("Nirjas statistics written to %s", stats_path)

            logger.info(
                "Nirjas dataset exported: %d total (%d train / %d test) → %s",
                result.nirjas_total,
                result.nirjas_train,
                result.nirjas_test,
                nirjas_path,
            )

        return result

    def print_summary(self, result: ExportResult) -> None:
        """Pretty-print the export results to stdout."""
        print("=" * 60)
        print("DATASET EXPORT SUMMARY")
        print("=" * 60)

        if result.atarashi_path:
            print("\n--- Atarashi (licence similarity) ---")
            print(f"  Path:           {result.atarashi_path}")
            print(f"  Total samples:  {result.atarashi_total:,}")
            print(f"  Train split:    {result.atarashi_train:,}")
            print(f"  Test split:     {result.atarashi_test:,}")
        else:
            print("\n  Atarashi: (not exported)")

        if result.nirjas_path:
            print("\n--- Nirjas (2-class classification) ---")
            print(f"  Path:           {result.nirjas_path}")
            print(f"  Total samples:  {result.nirjas_total:,}")
            print(f"  Train split:    {result.nirjas_train:,}")
            print(f"  Test split:     {result.nirjas_test:,}")
        else:
            print("\n  Nirjas: (not exported)")

        print("=" * 60)


# Convenience function


def export_datasets(
    atarashi_samples: Optional[list[AtarashiSample]] = None,
    nirjas_samples: Optional[list[NirjasSample]] = None,
    output_dir: str = "output",
    train_split_ratio: float = 0.8,
    random_seed: int = 42,
    write_statistics: bool = True,
) -> ExportResult:
    """One-shot convenience function for exporting both datasets.

    See :class:`DatasetExporter` for full parameter documentation.
    """
    config = ExportConfig(
        output_dir=output_dir,
        train_split_ratio=train_split_ratio,
        random_seed=random_seed,
        write_statistics=write_statistics,
    )
    exporter = DatasetExporter(config)
    return exporter.export(
        atarashi_samples=atarashi_samples,
        nirjas_samples=nirjas_samples,
    )
