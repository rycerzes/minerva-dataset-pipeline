from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fetchers.scancode import ScanCodeFetcher
from fetchers.fossology import FossologyFetcher
from builder.hybrid_merge import (
    HybridMerger,
    save_hybrid_dataset,
)


def run_pipeline(
    output_path: str = "hybrid_dataset.parquet",
    scancode_base_url: str = "https://scancode-licensedb.aboutcode.org",
    fossology_base_url: str = "https://raw.githubusercontent.com/fossology/fossology/master/install/db/licenseRef.json",
    include_exceptions: bool = False,
    include_deprecated: bool = False,
    verbose: bool = False,
) -> dict:
    print("=" * 60)
    print("Minerva Dataset Pipeline")
    print("=" * 60)

    print("\n[1/3] Fetching ScanCode LicenseDB...")
    scancode_fetcher = ScanCodeFetcher(base_url=scancode_base_url)
    scancode_licenses = scancode_fetcher.fetch_all(
        include_exceptions=include_exceptions,
        include_deprecated=include_deprecated,
    )
    scancode_count = len(scancode_licenses)
    scancode_with_text = sum(1 for lic in scancode_licenses if lic.license_text)
    print(f"  Fetched {scancode_count} licenses ({scancode_with_text} with text)")

    print("\n[2/3] Fetching FOSSology licenseRef.json...")
    fossology_fetcher = FossologyFetcher(base_url=fossology_base_url)
    fossology_licenses = fossology_fetcher.fetch_all()
    fossology_count = len(fossology_licenses)
    fossology_with_text = sum(1 for lic in fossology_licenses if lic.rf_text)
    print(f"  Fetched {fossology_count} licenses ({fossology_with_text} with text)")

    print("\n[3/3] Merging datasets...")
    merger = HybridMerger(scancode_licenses, fossology_licenses)
    dataset = merger.merge()
    stats = merger.get_statistics(dataset)

    print(f"  Total entries: {stats['total']}")
    print(f"  ScanCode: {stats['scancode']}")
    print(f"  FOSSology legacy: {stats['fossology_legacy']}")
    print(f"  Exceptions: {stats['exceptions']}")

    print(f"\nSaving to {output_path}...")
    count = save_hybrid_dataset(dataset, output_path)
    print(f"  Saved {count} entries")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    return {
        "scancode_fetched": scancode_count,
        "scancode_with_text": scancode_with_text,
        "fossology_fetched": fossology_count,
        "fossology_with_text": fossology_with_text,
        "output_path": output_path,
        "entries_written": count,
        "statistics": stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Minerva Dataset Pipeline - Generate training dataset for Atarashi"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="hybrid_dataset.parquet",
        help="Output path for the merged dataset (default: hybrid_dataset.parquet)",
    )
    parser.add_argument(
        "--scancode-url",
        default="https://scancode-licensedb.aboutcode.org",
        help="ScanCode LicenseDB base URL",
    )
    parser.add_argument(
        "--fossology-url",
        default="https://raw.githubusercontent.com/fossology/fossology/master/install/db/licenseRef.json",
        help="FOSSology licenseRef.json URL",
    )
    parser.add_argument(
        "--include-exceptions",
        action="store_true",
        help="Include license exceptions in the dataset",
    )
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated licenses in the dataset",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    run_pipeline(
        output_path=args.output,
        scancode_base_url=args.scancode_url,
        fossology_base_url=args.fossology_url,
        include_exceptions=args.include_exceptions,
        include_deprecated=args.include_deprecated,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
