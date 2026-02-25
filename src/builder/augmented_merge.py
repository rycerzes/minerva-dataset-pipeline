"""Synthetic Data Integration.

Integrates LLM-generated synthetic data with the real-world hybrid dataset
produced by the hybrid merger.  Responsibilities:

* **Provenance tracking** — every sample records whether it comes from
  ``scancode``, ``fossology``, or ``synthetic``.
* **Stratified sampling** for Atarashi (license-similarity training):
    - Over-represented classes are down-sampled to a configurable cap.
    - Under-represented classes are augmented with LLM-injected fragments.
    - The max imbalance ratio is kept within a configurable bound.
* **Nirjas integration** — accepts the already-balanced 2-class dataset
  from :mod:`augmentation.class_balancing` and converts it to export-ready
  ``NirjasSample`` records.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

try:
    from .hybrid_merge import DatasetEntry, DataSource
    from ..augmentation.legal_structure_splitter import SplitFragment
    from ..augmentation.llm_synthetic import AugmentedFragment
    from ..augmentation.hard_negative_generator import HardNegativeSample
    from ..augmentation.class_balancing import BalancedSample
except ImportError:  # pragma: no cover — direct-script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from builder.hybrid_merge import DatasetEntry, DataSource
    from augmentation.legal_structure_splitter import SplitFragment
    from augmentation.llm_synthetic import AugmentedFragment
    from augmentation.hard_negative_generator import HardNegativeSample
    from augmentation.class_balancing import BalancedSample

logger = logging.getLogger(__name__)


class AtarashiSample(BaseModel):
    """One row in the Atarashi licence-similarity dataset."""

    model_config = {"protected_namespaces": ()}

    license_key: str
    text: str
    source: str  # "scancode", "fossology", or "synthetic"


class NirjasSample(BaseModel):
    """One row in the Nirjas 2-class comment-classification dataset."""

    model_config = {"protected_namespaces": ()}

    text: str
    label: str  # "license_related" or "not_license_related"
    source: str  # "scancode", "fossology", "synthetic", or "llm_generated"
    negative_type: Optional[str] = None  # Class 2 only


class AugmentedMergerConfig(BaseModel):
    """Knobs for the augmented-merge step."""

    model_config = {"protected_namespaces": ()}

    min_samples_per_class: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum number of text samples every licence class must have in the "
            "Atarashi dataset.  Under-represented classes are padded with synthetic "
            "augmented fragments."
        ),
    )
    max_samples_per_class: Optional[int] = Field(
        default=None,
        description=(
            "Hard cap on samples per licence class.  Over-represented classes "
            "are down-sampled to this limit.  ``None`` = no cap."
        ),
    )
    max_imbalance_ratio: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Maximum allowed ratio between the largest and smallest class. "
            "If exceeded, the largest classes are down-sampled."
        ),
    )
    random_seed: int = 42


class AugmentedMerger:
    """Combine real-world + synthetic data into export-ready datasets.

    Inputs (all optional — supply whatever the pipeline has produced):

    * ``base_dataset`` — ``DatasetEntry`` list from :class:`HybridMerger`.
    * ``fragments`` — ``SplitFragment`` list from the sliding-window splitter.
    * ``augmented_fragments`` — ``AugmentedFragment`` list from the
      surgical-LLM injector.
    * ``hard_negatives`` — ``HardNegativeSample`` list from the hard-neg
      generator.
    * ``nirjas_balanced`` — ``BalancedSample`` list from
      :class:`NirjasClassBalancer`.
    """

    def __init__(self, config: Optional[AugmentedMergerConfig] = None):
        self.config = config or AugmentedMergerConfig()
        self._rng = random.Random(self.config.random_seed)

    def _collect_atarashi_samples(
        self,
        base_dataset: list[DatasetEntry],
        fragments: list[SplitFragment],
        augmented_fragments: list[AugmentedFragment],
    ) -> dict[str, list[AtarashiSample]]:
        """Group all candidate Atarashi samples by ``license_key``."""

        by_key: dict[str, list[AtarashiSample]] = defaultdict(list)
        seen: dict[str, set[str]] = defaultdict(set)  # dedup per key

        # 1. Full licence texts from the base hybrid dataset
        for entry in base_dataset:
            text = (entry.license_text or "").strip()
            if not text:
                continue
            if text in seen[entry.license_key]:
                continue
            seen[entry.license_key].add(text)
            by_key[entry.license_key].append(
                AtarashiSample(
                    license_key=entry.license_key,
                    text=text,
                    source=entry.source.value,
                )
            )

        # 2. Raw sliding-window fragments (source inherits from parent licence)
        for frag in fragments:
            text = frag.fragment_text.strip()
            if not text or text in seen[frag.license_key]:
                continue
            seen[frag.license_key].add(text)
            by_key[frag.license_key].append(
                AtarashiSample(
                    license_key=frag.license_key,
                    text=text,
                    source=frag.source,
                )
            )

        # 3. LLM-augmented fragments → marked as "synthetic"
        for aug in augmented_fragments:
            text = aug.augmented_text.strip()
            if not text or text in seen[aug.original_fragment.license_key]:
                continue
            # Skip pass-through (no placeholders were actually filled)
            if text == aug.original_fragment.fragment_text.strip():
                continue
            key = aug.original_fragment.license_key
            seen[key].add(text)
            by_key[key].append(
                AtarashiSample(
                    license_key=key,
                    text=text,
                    source="synthetic",
                )
            )

        return dict(by_key)

    def _stratify(
        self,
        by_key: dict[str, list[AtarashiSample]],
    ) -> list[AtarashiSample]:
        """Apply stratified sampling to honour class-balance constraints.

        Strategy
        --------
        1. If ``max_samples_per_class`` is set, down-sample every class that
           exceeds the cap.
        2. Up-sample (duplicate) every class that has fewer than
           ``min_samples_per_class`` samples.
        3. If the resulting imbalance ratio exceeds
           ``max_imbalance_ratio``, progressively down-sample the largest
           classes until the ratio is within bounds.
        """
        if not by_key:
            return []

        cfg = self.config
        pool: dict[str, list[AtarashiSample]] = {}

        # Step 1: apply hard cap
        for key, samples in by_key.items():
            if cfg.max_samples_per_class is not None and len(samples) > cfg.max_samples_per_class:
                pool[key] = self._rng.sample(samples, cfg.max_samples_per_class)
            else:
                pool[key] = list(samples)

        # Step 2: pad under-represented classes
        for key, samples in pool.items():
            if len(samples) < cfg.min_samples_per_class:
                pool[key] = self._upsample(samples, cfg.min_samples_per_class)

        # Step 3: enforce imbalance ratio
        pool = self._enforce_imbalance_ratio(pool)

        # Flatten + shuffle
        result: list[AtarashiSample] = []
        for samples in pool.values():
            result.extend(samples)
        self._rng.shuffle(result)
        return result

    def _upsample(
        self,
        samples: list[AtarashiSample],
        target: int,
    ) -> list[AtarashiSample]:
        """Duplicate random items from *samples* until *target* is reached."""
        if not samples:
            return []
        result = list(samples)
        while len(result) < target:
            result.append(self._rng.choice(samples))
        return result[:target]

    def _enforce_imbalance_ratio(
        self,
        pool: dict[str, list[AtarashiSample]],
    ) -> dict[str, list[AtarashiSample]]:
        """Down-sample the largest classes so the imbalance ratio stays within bounds."""
        if not pool:
            return pool

        max_ratio = self.config.max_imbalance_ratio
        counts = {k: len(v) for k, v in pool.items()}
        min_count = min(counts.values())

        if min_count == 0:
            return pool

        cap = max(1, int(min_count * max_ratio))

        for key in pool:
            if len(pool[key]) > cap:
                pool[key] = self._rng.sample(pool[key], cap)

        return pool

    @staticmethod
    def _convert_nirjas(
        nirjas_balanced: list[BalancedSample],
    ) -> list[NirjasSample]:
        """Convert pre-balanced ``BalancedSample`` objects to ``NirjasSample``."""
        return [
            NirjasSample(
                text=s.text,
                label=s.label,
                source=s.source,
                negative_type=s.negative_type,
            )
            for s in nirjas_balanced
            if s.text.strip()
        ]

    def merge(
        self,
        base_dataset: Optional[list[DatasetEntry]] = None,
        fragments: Optional[list[SplitFragment]] = None,
        augmented_fragments: Optional[list[AugmentedFragment]] = None,
        hard_negatives: Optional[list[HardNegativeSample]] = None,
        nirjas_balanced: Optional[list[BalancedSample]] = None,
    ) -> tuple[list[AtarashiSample], list[NirjasSample]]:
        """Run the augmented merge and return both datasets.

        Parameters
        ----------
        base_dataset:
            Hybrid-merged ``DatasetEntry`` objects.
        fragments:
            Sliding-window ``SplitFragment`` objects.
        augmented_fragments:
            LLM-injected ``AugmentedFragment`` objects.
        hard_negatives:
            Hard-negative ``HardNegativeSample`` objects.
            Currently unused directly — they should be fed through
            :class:`NirjasClassBalancer` first and passed as
            *nirjas_balanced*.  Accepted here for forward-compatibility.
        nirjas_balanced:
            Balanced ``BalancedSample`` objects.

        Returns
        -------
        (atarashi_samples, nirjas_samples)
        """
        base_dataset = base_dataset or []
        fragments = fragments or []
        augmented_fragments = augmented_fragments or []
        nirjas_balanced = nirjas_balanced or []

        # --- Atarashi ---
        by_key = self._collect_atarashi_samples(
            base_dataset, fragments, augmented_fragments
        )

        logger.info(
            "Atarashi pre-stratification: %d classes, %d total samples",
            len(by_key),
            sum(len(v) for v in by_key.values()),
        )

        atarashi = self._stratify(by_key)

        logger.info("Atarashi post-stratification: %d samples", len(atarashi))

        # --- Nirjas ---
        nirjas = self._convert_nirjas(nirjas_balanced)

        logger.info("Nirjas samples: %d", len(nirjas))

        return atarashi, nirjas

    @staticmethod
    def get_atarashi_statistics(samples: list[AtarashiSample]) -> dict:
        """Return summary statistics for the Atarashi dataset."""
        if not samples:
            return {
                "total": 0,
                "unique_licenses": 0,
                "by_source": {},
                "min_per_class": 0,
                "max_per_class": 0,
                "imbalance_ratio": 0.0,
            }

        by_source: dict[str, int] = {}
        by_key: dict[str, int] = {}

        for s in samples:
            by_source[s.source] = by_source.get(s.source, 0) + 1
            by_key[s.license_key] = by_key.get(s.license_key, 0) + 1

        counts = list(by_key.values())
        min_c = min(counts)
        max_c = max(counts)

        return {
            "total": len(samples),
            "unique_licenses": len(by_key),
            "by_source": by_source,
            "min_per_class": min_c,
            "max_per_class": max_c,
            "imbalance_ratio": max_c / min_c if min_c > 0 else float("inf"),
        }

    @staticmethod
    def get_nirjas_statistics(samples: list[NirjasSample]) -> dict:
        """Return summary statistics for the Nirjas dataset."""
        if not samples:
            return {
                "total": 0,
                "license_related": 0,
                "not_license_related": 0,
                "actual_ratio": 0.0,
                "by_source": {},
                "by_negative_type": {},
            }

        n_lic = sum(1 for s in samples if s.label == "license_related")
        n_neg = sum(1 for s in samples if s.label == "not_license_related")
        by_source: dict[str, int] = {}
        by_neg_type: dict[str, int] = {}

        for s in samples:
            by_source[s.source] = by_source.get(s.source, 0) + 1
            if s.negative_type:
                by_neg_type[s.negative_type] = (
                    by_neg_type.get(s.negative_type, 0) + 1
                )

        return {
            "total": len(samples),
            "license_related": n_lic,
            "not_license_related": n_neg,
            "actual_ratio": n_lic / len(samples) if samples else 0.0,
            "by_source": by_source,
            "by_negative_type": by_neg_type,
        }

    def print_statistics(
        self,
        atarashi: list[AtarashiSample],
        nirjas: list[NirjasSample],
    ) -> None:
        """Pretty-print a combined report to stdout."""
        a_stats = self.get_atarashi_statistics(atarashi)
        n_stats = self.get_nirjas_statistics(nirjas)

        print("=" * 60)
        print("AUGMENTED MERGE REPORT")
        print("=" * 60)

        print("\n--- Atarashi (licence similarity) ---")
        print(f"  Total samples:        {a_stats['total']:,}")
        print(f"  Unique licences:      {a_stats['unique_licenses']:,}")
        print(f"  Min samples/class:    {a_stats['min_per_class']:,}")
        print(f"  Max samples/class:    {a_stats['max_per_class']:,}")
        print(f"  Imbalance ratio:      {a_stats['imbalance_ratio']:.2f}x")
        print("  By source:")
        for src, count in sorted(a_stats["by_source"].items()):
            print(f"    {src:20s} {count:,}")

        print("\n--- Nirjas (2-class classification) ---")
        print(f"  Total samples:        {n_stats['total']:,}")
        print(f"  license_related:      {n_stats['license_related']:,}")
        print(f"  not_license_related:  {n_stats['not_license_related']:,}")
        print(f"  Actual ratio:         {n_stats['actual_ratio']:.4f}")
        print("  By source:")
        for src, count in sorted(n_stats["by_source"].items()):
            print(f"    {src:20s} {count:,}")
        if n_stats["by_negative_type"]:
            print("  Negative types:")
            for ntype, count in sorted(n_stats["by_negative_type"].items()):
                print(f"    {ntype:20s} {count:,}")

        print("=" * 60)


def augmented_merge(
    base_dataset: Optional[list[DatasetEntry]] = None,
    fragments: Optional[list[SplitFragment]] = None,
    augmented_fragments: Optional[list[AugmentedFragment]] = None,
    hard_negatives: Optional[list[HardNegativeSample]] = None,
    nirjas_balanced: Optional[list[BalancedSample]] = None,
    min_samples_per_class: int = 3,
    max_samples_per_class: Optional[int] = None,
    max_imbalance_ratio: float = 10.0,
    random_seed: int = 42,
) -> tuple[list[AtarashiSample], list[NirjasSample]]:
    """One-shot convenience function for the augmented merge.

    See :class:`AugmentedMerger` for full parameter documentation.
    """
    config = AugmentedMergerConfig(
        min_samples_per_class=min_samples_per_class,
        max_samples_per_class=max_samples_per_class,
        max_imbalance_ratio=max_imbalance_ratio,
        random_seed=random_seed,
    )
    merger = AugmentedMerger(config)
    return merger.merge(
        base_dataset=base_dataset,
        fragments=fragments,
        augmented_fragments=augmented_fragments,
        hard_negatives=hard_negatives,
        nirjas_balanced=nirjas_balanced,
    )
