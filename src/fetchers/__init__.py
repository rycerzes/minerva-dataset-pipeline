from .scancode import (
    ScanCodeFetcher,
    LicenseData,
    LicenseDetails,
    fetch_scancode_licenses,
    fetch_and_save_parquet as fetch_scancode_parquet,
)

from .fossology import (
    FossologyFetcher,
    FossologyLicense,
    fetch_fossology_licenses,
    fetch_and_save_parquet as fetch_fossology_parquet,
)

from ..utils import model_to_parquet, iter_to_parquet, read_parquet

__all__ = [
    "ScanCodeFetcher",
    "LicenseData",
    "LicenseDetails",
    "fetch_scancode_licenses",
    "fetch_scancode_parquet",
    "FossologyFetcher",
    "FossologyLicense",
    "fetch_fossology_licenses",
    "fetch_fossology_parquet",
    "model_to_parquet",
    "iter_to_parquet",
    "read_parquet",
]
