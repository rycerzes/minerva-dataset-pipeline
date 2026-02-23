from __future__ import annotations

import httpx
from pydantic import BaseModel, Field
from typing import Optional

from ..utils import model_to_parquet


LICENSEDB_BASE_URL = "https://scancode-licensedb.aboutcode.org"


class LicenseIndexEntry(BaseModel):
    model_config = {"protected_namespaces": ()}

    license_key: str
    category: str
    spdx_license_key: Optional[str] = None
    other_spdx_license_keys: list[str] = Field(default_factory=list)
    is_exception: bool = False
    is_deprecated: bool = False
    json: str
    yaml: str
    html: str
    license: str


class LicenseDetails(BaseModel):
    key: str
    short_name: str
    name: str
    category: str
    owner: Optional[str] = None
    homepage_url: Optional[str] = None
    notes: Optional[str] = None
    spdx_license_key: Optional[str] = None
    other_spdx_license_keys: list[str] = Field(default_factory=list)
    osi_license_key: Optional[str] = None
    text_urls: list[str] = Field(default_factory=list)
    osi_url: Optional[str] = None
    faq_url: Optional[str] = None
    other_urls: list[str] = Field(default_factory=list)
    ignorable_urls: list[str] = Field(default_factory=list)
    text: Optional[str] = None


class LicenseData(BaseModel):
    license_key: str
    short_name: str
    name: str
    category: str
    spdx_license_key: Optional[str] = None
    is_exception: bool = False
    is_deprecated: bool = False
    license_text: Optional[str] = None
    source_url: Optional[str] = None


class ScanCodeFetcher:
    def __init__(self, base_url: str = LICENSEDB_BASE_URL):
        self.base_url = base_url
        self._client: Optional[httpx.Client] = None
        self._index: Optional[list[LicenseIndexEntry]] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "ScanCodeFetcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def fetch_index(self, force: bool = False) -> list[LicenseIndexEntry]:
        if self._index is not None and not force:
            return self._index

        url = f"{self.base_url}/index.json"
        response = self.client.get(url)
        response.raise_for_status()

        data = response.json()
        self._index = [LicenseIndexEntry.model_validate(entry) for entry in data]
        return self._index

    def fetch_license_details(self, license_json_path: str) -> LicenseDetails:
        url = f"{self.base_url}/{license_json_path}"
        response = self.client.get(url)
        response.raise_for_status()

        data = response.json()
        return LicenseDetails.model_validate(data)

    def fetch_all_licenses(
        self,
        include_exceptions: bool = False,
        include_deprecated: bool = False,
    ) -> list[LicenseData]:
        index_entries = self.fetch_index()

        licenses: list[LicenseData] = []

        for entry in index_entries:
            if entry.is_exception and not include_exceptions:
                continue
            if entry.is_deprecated and not include_deprecated:
                continue

            try:
                details = self.fetch_license_details(entry.json)
            except httpx.HTTPError:
                continue

            license_data = LicenseData(
                license_key=details.key,
                short_name=details.short_name,
                name=details.name,
                category=details.category,
                spdx_license_key=details.spdx_license_key,
                is_exception=entry.is_exception,
                is_deprecated=entry.is_deprecated,
                license_text=details.text,
                source_url=f"{self.base_url}/{entry.json}",
            )
            licenses.append(license_data)

        return licenses

    def iter_licenses(
        self,
        include_exceptions: bool = False,
        include_deprecated: bool = False,
    ):
        index_entries = self.fetch_index()

        for entry in index_entries:
            if entry.is_exception and not include_exceptions:
                continue
            if entry.is_deprecated and not include_deprecated:
                continue

            try:
                details = self.fetch_license_details(entry.json)
            except httpx.HTTPError:
                continue

            yield LicenseData(
                license_key=details.key,
                short_name=details.short_name,
                name=details.name,
                category=details.category,
                spdx_license_key=details.spdx_license_key,
                is_exception=entry.is_exception,
                is_deprecated=entry.is_deprecated,
                license_text=details.text,
                source_url=f"{self.base_url}/{entry.json}",
            )


def fetch_scancode_licenses(
    include_exceptions: bool = False,
    include_deprecated: bool = False,
) -> list[LicenseData]:
    with ScanCodeFetcher() as fetcher:
        return fetcher.fetch_all_licenses(
            include_exceptions=include_exceptions,
            include_deprecated=include_deprecated,
        )


def fetch_and_save_parquet(
    output_path: str,
    include_exceptions: bool = False,
    include_deprecated: bool = False,
) -> int:
    licenses = fetch_scancode_licenses(
        include_exceptions=include_exceptions,
        include_deprecated=include_deprecated,
    )
    model_to_parquet(licenses, output_path)
    return len(licenses)


if __name__ == "__main__":
    with ScanCodeFetcher() as fetcher:
        index = fetcher.fetch_index()
        print(f"Fetched {len(index)} license entries from ScanCode LicenseDB")

        sample = fetcher.fetch_license_details(index[0].json)
        print(f"Sample license: {sample.short_name}")
