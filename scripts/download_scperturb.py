"""
Download scPerturb datasets from Zenodo with resume, checksum, and parallelism.

Defaults to the RNA+protein record (10.5281/zenodo.13350497) and filters to CRISPR-like
perturbations only. You can tweak the whitelist/blacklist patterns below or pass your own.

Examples
--------
# Dry-run (list what would be downloaded)
python download_scperturb.py --dry-run -o /data/scperturb

# Download CRISPR-only with 8 workers
python download_scperturb.py -o /data/scperturb --workers 8

# Download everything in the record
python download_scperturb.py -o /data/scperturb --include-all

# Override filters (comma-separated substrings)
python download_scperturb.py -o /data/scperturb \
  --include "DixitRegev,NormanWeissman" --exclude "sciplex,McFarland,ZhaoSims"

Notes
-----
- Uses Zenodo REST API to enumerate files and md5 checksums.
- Resumes partial downloads when the server advertises 'Accept-Ranges: bytes'.
- Verifies md5 digest after download (or skip with --no-check)
- Writes a manifest.json with names, sizes, md5, and status.

"""

import concurrent.futures as cf
import contextlib
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
import time

import beartype
import requests
import tyro

ZENODO_RECORD_DEFAULT = "13350497"  # scPerturb RNA+protein h5ad files
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "scperturb-downloader/1.0"})

# Conservative defaults:
# - Whitelist clearly CRISPR-based datasets by substring
CRISPR_WHITELIST = [
    "AdamsonWeissman2016",
    "DixitRegev2016",
    "GasperiniShendure2019",
    "NormanWeissman2019",
    "ShifrutMarson2018",
    "TianKampmann2019",
    "TianKampmann2021_CRISPRa",
    "TianKampmann2021_CRISPRi",
    "JoungZhang2023",
    "SchraivogelSteinmetz2020",
    "XieHon2017",
    "FrangiehIzar2021",
]
# - Blacklist known drug datasets by substring
DRUG_BLACKLIST = [
    "SrivatsanTrapnell2020_sciplex",
    "McFarlandTsherniak2020",
    "ZhaoSims2021",
    "AissaBenevolenskaya2021",
]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FileEntry:
    key: str
    size: int
    checksum: str | None
    url: str

    @property
    def md5(self) -> str | None:
        if not self.checksum:
            return None
        # Zenodo checksum format is usually 'md5:abcdef...'
        if ":" in self.checksum:
            algo, hexd = self.checksum.split(":", 1)
            if algo.lower() == "md5":
                return hexd
        # Fallback: assume already hex
        return self.checksum


@beartype.beartype
def list_record_files(record_id: str) -> list[FileEntry]:
    """Return list of files for a published Zenodo record.

    Handles both legacy and InvenioRDM-style JSON shapes.
    """
    api = f"https://zenodo.org/api/records/{record_id}"
    r = SESSION.get(api, timeout=60)
    r.raise_for_status()
    data = r.json()

    entries: list[FileEntry] = []
    files_block = data.get("files")

    def add_entry(key: str, entry: dict):
        size = entry.get("size") or entry.get("filesize") or 0
        chksum = entry.get("checksum")
        links = entry.get("links", {})
        url = (
            links.get("content")
            or links.get("self")
            or f"https://zenodo.org/records/{record_id}/files/{key}?download=1"
        )
        entries.append(FileEntry(key=key, size=int(size), checksum=chksum, url=url))

    if isinstance(files_block, dict) and "entries" in files_block:
        for key, entry in files_block["entries"].items():
            add_entry(key, entry)
    elif isinstance(files_block, list):
        for entry in files_block:
            key = entry.get("key") or entry.get("filename")
            if not key:
                continue
            add_entry(key, entry)
    else:
        # Fall back to scraping the HTML file list if API schema changes
        # (not ideal, but gives a usable URL template)
        # Filenames are stable, so we construct URLs directly.
        # In this mode we won't have checksums/sizes.
        print(
            "Warning: unexpected API shape for 'files'; falling back to name-only.",
            file=sys.stderr,
        )
        # try a minimal list by hitting the HTML and regexing .h5ad names
        html = SESSION.get(f"https://zenodo.org/records/{record_id}", timeout=60).text
        for m in re.finditer(r">([A-Za-z0-9_][^\s<>]*?\.h5ad)<", html):
            key = m.group(1)
            url = f"https://zenodo.org/records/{record_id}/files/{key}?download=1"
            entries.append(FileEntry(key=key, size=0, checksum=None, url=url))

    return entries


@beartype.beartype
def choose_files(
    files: list[FileEntry],
    include_all: bool,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[FileEntry]:
    if include_all:
        candidates = files
    else:
        # Start with whitelist
        wl = include_patterns or CRISPR_WHITELIST
        candidates = [f for f in files if any(p in f.key for p in wl)]
    # Apply blacklist
    bl = exclude_patterns or DRUG_BLACKLIST
    filtered = [f for f in candidates if not any(p in f.key for p in bl)]
    return filtered


@beartype.beartype
def md5sum(path: pathlib.Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@beartype.beartype
def head(url: str) -> tuple[int, bool]:
    """Return (size, supports_range)."""
    with contextlib.suppress(Exception):
        resp = SESSION.head(url, timeout=60, allow_redirects=True)
        size = int(resp.headers.get("Content-Length", "0"))
        accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
        return size, accept_ranges
    return 0, False


@beartype.beartype
def download_one(
    entry: FileEntry,
    outdir: pathlib.Path,
    retries: int = 5,
    verify_md5: bool = True,
    progress: bool = True,
) -> dict:
    dest = outdir / entry.key
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size, supports_range = head(entry.url)
    total = entry.size or remote_size

    resume_pos = dest.stat().st_size if dest.exists() else 0
    if total and resume_pos >= total:
        status = "exists"
    else:
        status = "downloaded"
        # Stream download (with resume if possible)
        for attempt in range(retries):
            headers = {}
            if supports_range and resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
            try:
                with SESSION.get(
                    entry.url, stream=True, timeout=120, headers=headers
                ) as r:
                    r.raise_for_status()
                    mode = "ab" if headers.get("Range") else "wb"
                    with dest.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                resume_pos += len(chunk)
                break
            except Exception as e:
                wait = min(60, 2**attempt)
                print(
                    f"Retry {attempt + 1}/{retries} for {entry.key} after error: {e} (sleep {wait}s)"
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"Failed to download {entry.key} after {retries} retries"
            )

    ok = True
    digest = None
    if verify_md5 and entry.md5:
        digest = md5sum(dest)
        ok = digest == entry.md5
        if not ok:
            # If mismatch and server supports range, attempt one clean retry
            dest.unlink(missing_ok=True)
            return download_one(entry, outdir, retries=retries, verify_md5=verify_md5)

    return {
        "name": entry.key,
        "size": dest.stat().st_size if dest.exists() else 0,
        "expected_md5": entry.md5,
        "actual_md5": digest,
        "ok": ok,
        "status": status,
        "url": entry.url,
    }


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Download scPerturb datasets from Zenodo."""

    out: pathlib.Path
    """Output directory"""

    record_id: str = ZENODO_RECORD_DEFAULT
    """Zenodo record ID"""

    include_all: bool = False
    """Download all files in the record"""

    include: str | None = None
    """Comma-separated substrings to INCLUDE (overrides builtin whitelist)"""

    exclude: str | None = None
    """Comma-separated substrings to EXCLUDE (overrides builtin blacklist)"""

    workers: int = 8
    """Parallel downloads"""

    no_check: bool = False
    """Skip md5 verification"""

    dry_run: bool = False
    """List planned downloads then exit"""


def main(cfg: Config):
    outdir = cfg.out
    outdir.mkdir(parents=True, exist_ok=True)

    files = list_record_files(cfg.record_id)
    include_patterns = (
        [s.strip() for s in cfg.include.split(",")] if cfg.include else []
    )
    exclude_patterns = (
        [s.strip() for s in cfg.exclude.split(",")] if cfg.exclude else []
    )

    files_to_get = choose_files(
        files,
        include_all=cfg.include_all,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    print(f"Found {len(files)} files in record {cfg.record_id}.")
    print(f"Selected {len(files_to_get)} to download.")
    for f in files_to_get:
        size_mb = (f.size or 0) / (1024 * 1024)
        print(f"  - {f.key}  ({size_mb:.1f} MB)")

    if cfg.dry_run:
        return 0

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = [
            ex.submit(download_one, f, outdir, verify_md5=not cfg.no_check)
            for f in files_to_get
        ]
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            status = "OK" if res["ok"] else "BAD-CHECKSUM"
            print(f"[{status}] {res['name']} ({res['size'] / 1024 / 1024:.1f} MB)")

    manifest = {
        "record_id": cfg.record_id,
        "outdir": str(outdir),
        "count": len(results),
        "results": results,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {outdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
