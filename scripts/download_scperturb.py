"""
Download scPerturb datasets from Zenodo with resume, checksum, and parallelism.

Changes in this version:
- Defaults to ALL scPerturb datasets (RNA+protein + ATAC) across multiple records.
- Queries Zenodo API to enumerate files and md5 (no hardcoded filenames).
- Blacklist-based filtering (optional). Whitelist optional, but default is include-all.
- Uses atomic *.part files, resume via Range, md5 verification, and bounded re-downloads.
- Renames existing mismatched files to *.corrupted (with numeric suffix) before re-downloading.
- Per-record subdirectories by default to avoid filename collisions.
- Summarized reporting + manifest.json; nonzero exit if failures.

Examples
--------
# Dry-run (list what would be downloaded)
python download_scperturb.py --dry-run -o /data/scperturb

# Download everything from both records with 8 workers
python download_scperturb.py -o /data/scperturb --workers 8

# Exclude specific patterns
python download_scperturb.py -o /data/scperturb --exclude sciplex McFarland

# Only RNA+protein record, flat layout, and no checksum verification
python download_scperturb.py -o /data/scperturb --records 13350497 --flat --no-check
"""

import collections.abc
import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import pathlib
import re
import time

import beartype
import requests
import tqdm
import tyro

# Default to both major scPerturb records:
# - 13350497: RNA+protein .h5ad bundle
# - 7058382 : ATAC archives
RECORD_IDS_DEFAULT = ["13350497", "7058382"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "scperturb-downloader/2.0"})

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("scperturb")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FileEntry:
    record_id: str
    key: str
    size: int
    checksum: str | None
    url: str

    @property
    def md5(self) -> str | None:
        if not self.checksum:
            return None
        if ":" in self.checksum:
            algo, hexd = self.checksum.split(":", 1)
            if algo.lower() == "md5":
                return hexd
        return self.checksum


@beartype.beartype
def _extract_files_from_record_json(record_id: str, data: dict) -> list[FileEntry]:
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
        entries.append(
            FileEntry(
                record_id=record_id,
                key=key,
                size=int(size) if size else 0,
                checksum=chksum,
                url=url,
            )
        )

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
        # Fallback: minimally parse HTML to discover filenames (size/md5 may be missing)
        html = SESSION.get(f"https://zenodo.org/records/{record_id}", timeout=60).text
        for m in re.finditer(r">([A-Za-z0-9_][^\s<>]*?\.(h5ad|zip|tar\.gz))<", html):
            key = m.group(1)
            url = f"https://zenodo.org/records/{record_id}/files/{key}?download=1"
            entries.append(
                FileEntry(record_id=record_id, key=key, size=0, checksum=None, url=url)
            )
    return entries


@beartype.beartype
def list_record_files(record_id: str) -> list[FileEntry]:
    api = f"https://zenodo.org/api/records/{record_id}"
    r = SESSION.get(api, timeout=60)
    r.raise_for_status()
    data = r.json()
    return _extract_files_from_record_json(record_id, data)


@beartype.beartype
def list_records_files(record_ids: collections.abc.Iterable[str]) -> list[FileEntry]:
    all_entries: list[FileEntry] = []
    for rid in record_ids:
        all_entries.extend(list_record_files(rid))
    return all_entries


@beartype.beartype
def _compile_patterns(patterns: collections.abc.Iterable[str]) -> list[re.Pattern]:
    compiled: list[re.Pattern] = []
    for p in patterns:
        # Treat as substring by default; if it looks like a regex, allow it.
        # We compile as regex either way (re.escape for pure substring handled below).
        try:
            compiled.append(re.compile(p))
        except re.error:
            compiled.append(re.compile(re.escape(p)))
    return compiled


@beartype.beartype
def choose_files(
    files: list[FileEntry],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[FileEntry]:
    keep = files
    if include_patterns:
        inc = _compile_patterns(include_patterns)
        keep = [f for f in keep if any(p.search(f.key) for p in inc)]
    if exclude_patterns:
        exc = _compile_patterns(exclude_patterns)
        keep = [f for f in keep if not any(p.search(f.key) for p in exc)]
    return keep


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
    try:
        resp = SESSION.head(url, timeout=60, allow_redirects=True)
        size = int(resp.headers.get("Content-Length", "0"))
        accept_ranges = "bytes" in resp.headers.get("Accept-Ranges", "").lower()
        return size, accept_ranges
    except requests.RequestException as e:
        logger.debug(f"HEAD request failed for {url}: {e}")
        return 0, False
    except Exception as e:
        logger.warning(f"Unexpected error in HEAD request for {url}: {e}")
        return 0, False


@beartype.beartype
def _unique_corrupted_path(path: pathlib.Path) -> pathlib.Path:
    base = pathlib.Path(str(path) + ".corrupted")
    if not base.exists():
        return base
    idx = 1
    while True:
        cand = pathlib.Path(str(path) + f".corrupted.{idx}")
        if not cand.exists():
            return cand
        idx += 1


@beartype.beartype
def download_one(
    entry: FileEntry,
    dest: pathlib.Path,
    *,
    net_retries: int,
    verify_md5: bool,
    max_redownloads: int,
    show_progress: bool,
) -> dict:
    """
    Return dict with fields:
      name, record_id, size, expected_md5, actual_md5, ok (bool), status, url
    status ∈ {"downloaded","redownloaded","skip-exists","bad-checksum","failed"}
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = pathlib.Path(str(dest) + ".part")

    remote_size, supports_range = head(entry.url)
    expected_md5 = entry.md5

    # If final file exists, verify and skip or rename to .corrupted on mismatch
    if dest.exists():
        if verify_md5 and expected_md5:
            actual = md5sum(dest)
            if actual == expected_md5:
                return {
                    "record_id": entry.record_id,
                    "name": entry.key,
                    "size": dest.stat().st_size,
                    "expected_md5": expected_md5,
                    "actual_md5": actual,
                    "ok": True,
                    "status": "skip-exists",
                    "url": entry.url,
                }
            else:
                corr = _unique_corrupted_path(dest)
                dest.rename(corr)
                logger.warning(
                    f"MD5 mismatch for existing {dest.name}; moved to {corr.name}"
                )
        else:
            # No verification requested or no MD5 supplied: treat as good and skip
            return {
                "record_id": entry.record_id,
                "name": entry.key,
                "size": dest.stat().st_size,
                "expected_md5": expected_md5,
                "actual_md5": None if verify_md5 else "skipped",
                "ok": True,
                "status": "skip-exists",
                "url": entry.url,
            }

    status = "downloaded"
    bytes_discarded = 0

    # Attempt bounded re-downloads on MD5 mismatch
    for redl in range(max_redownloads):
        # Resume if *.part exists and server supports it
        resume_pos = part.stat().st_size if part.exists() else 0
        headers = {}
        if supports_range and resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"

        # Network retries with backoff
        for attempt in range(net_retries):
            try:
                with SESSION.get(
                    entry.url, stream=True, timeout=120, headers=headers
                ) as r:
                    r.raise_for_status()
                    mode = "ab" if "Range" in headers else "wb"
                    total = entry.size or remote_size or None
                    pbar = None
                    if show_progress:
                        pbar = tqdm.tqdm(
                            total=total,
                            initial=resume_pos,
                            unit="B",
                            unit_scale=True,
                            desc=f"{entry.record_id}/{entry.key}",
                            leave=False,
                        )
                    with part.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                if pbar:
                                    pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                break  # success, exit net retry loop
            except Exception as e:
                wait = min(60, 2**attempt)
                logger.warning(
                    f"{entry.key}: network error (attempt {attempt + 1}/{net_retries}): {e}; sleeping {wait}s"
                )
                time.sleep(wait)
        else:
            # Exhausted net retries
            if part.exists():
                bytes_discarded += part.stat().st_size
                part.unlink(missing_ok=True)
            return {
                "record_id": entry.record_id,
                "name": entry.key,
                "size": 0,
                "expected_md5": expected_md5,
                "actual_md5": None,
                "ok": False,
                "status": "failed",
                "url": entry.url,
                "bytes_discarded": bytes_discarded,
            }

        # Verify MD5 on the assembled *.part, then finalize
        actual = None
        ok = True
        if verify_md5 and expected_md5:
            actual = md5sum(part)
            ok = actual == expected_md5

        if ok:
            # Replace any leftover dest (shouldn't exist after rename branch above)
            if dest.exists():
                dest.unlink(missing_ok=True)
            part.replace(dest)
            if redl > 0:
                status = "redownloaded"
            return {
                "record_id": entry.record_id,
                "name": entry.key,
                "size": dest.stat().st_size,
                "expected_md5": expected_md5,
                "actual_md5": actual,
                "ok": True,
                "status": status,
                "url": entry.url,
                "bytes_discarded": bytes_discarded,
            }
        else:
            # MD5 mismatch: discard *.part and try again (bounded)
            logger.warning(
                f"MD5 mismatch for {entry.key} (attempt {redl + 1}/{max_redownloads}); retrying"
            )
            if part.exists():
                bytes_discarded += part.stat().st_size
                part.unlink(missing_ok=True)

    # Exceeded max redownload attempts
    return {
        "record_id": entry.record_id,
        "name": entry.key,
        "size": 0,
        "expected_md5": expected_md5,
        "actual_md5": None,
        "ok": False,
        "status": "bad-checksum",
        "url": entry.url,
    }


@beartype.beartype
def _load_patterns_file(path: pathlib.Path | None) -> list[str]:
    if not path:
        return []
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Download scPerturb datasets from Zenodo."""

    out: pathlib.Path
    """Output directory."""

    records: list[str] = dataclasses.field(default_factory=lambda: RECORD_IDS_DEFAULT)
    """Zenodo record IDs."""

    include: list[str] = dataclasses.field(default_factory=list)
    """Regex/substring patterns to INCLUDE."""

    exclude: list[str] = dataclasses.field(default_factory=list)
    """Regex/substring patterns to EXCLUDE."""

    exclude_file: pathlib.Path | None = None
    """Optional newline-delimited patterns file to EXCLUDE."""

    workers: int = 8
    """Parallel downloads."""

    no_check: bool = False
    """Skip md5 verification."""

    dry_run: bool = False
    """List planned downloads then exit."""

    flat: bool = False
    """Write files directly under out/ (no per-record subdirs)."""

    max_redownloads: int = 3
    """Max times to re-download a file after failing MD5 verification."""

    net_retries: int = 5
    """Retries per network GET attempt (with backoff)."""

    progress: bool = True
    """Show per-file progress bars."""


def main(cfg: Config) -> int:
    outdir = cfg.out
    outdir.mkdir(parents=True, exist_ok=True)

    # Enumerate all files across records
    files = list_records_files(cfg.records)

    # Compose filters
    exclude_patterns = cfg.exclude + _load_patterns_file(cfg.exclude_file)
    files_to_get = choose_files(
        files, include_patterns=cfg.include, exclude_patterns=exclude_patterns
    )

    # Present plan
    print(
        f"Found {len(files)} files across {len(cfg.records)} record(s): {', '.join(cfg.records)}"
    )
    print(f"Selected {len(files_to_get)} to download.\n")
    for f in files_to_get:
        size_mb = (f.size or 0) / (1024 * 1024)
        rel = pathlib.Path(f.record_id, f.key) if not cfg.flat else pathlib.Path(f.key)
        print(f"  - {rel}  ({size_mb:.1f} MB)")

    if cfg.dry_run:
        return 0

    # Download
    results: list[dict] = []
    reused_bytes = 0
    downloaded_bytes = 0
    discarded_bytes = 0
    failures = 0

    def _submit(entry: FileEntry):
        dest = (
            (outdir / entry.key) if cfg.flat else (outdir / entry.record_id / entry.key)
        )
        res = download_one(
            entry,
            dest,
            net_retries=cfg.net_retries,
            verify_md5=not cfg.no_check,
            max_redownloads=cfg.max_redownloads,
            show_progress=cfg.progress,
        )
        return res

    with cf.ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = [ex.submit(_submit, f) for f in files_to_get]
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            status = res["status"]
            name = (
                f"{res.get('record_id')}/{res.get('name')}"
                if not cfg.flat
                else res.get("name")
            )
            size_mb = (res.get("size", 0) or 0) / (1024 * 1024)
            if status == "skip-exists":
                reused_bytes += res.get("size", 0) or 0
                print(f"[SKIP-EXISTS] {name} ({size_mb:.1f} MB)")
            elif status in ("downloaded", "redownloaded"):
                downloaded_bytes += res.get("size", 0) or 0
                print(f"[OK] {name} ({size_mb:.1f} MB)")
            elif status == "bad-checksum":
                failures += 1
                discarded_bytes += res.get("bytes_discarded", 0) or 0
                print(f"[BAD-CHECKSUM] {name}")
            else:
                failures += 1
                discarded_bytes += res.get("bytes_discarded", 0) or 0
                print(f"[FAILED] {name}")

    # Summary + manifest
    summary = {
        "records": cfg.records,
        "outdir": str(outdir),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "total": len(results),
            "ok": sum(1 for r in results if r["ok"]),
            "skip_exists": sum(1 for r in results if r["status"] == "skip-exists"),
            "bad_checksum": sum(1 for r in results if r["status"] == "bad-checksum"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
        },
        "bytes": {
            "downloaded": downloaded_bytes,
            "reused": reused_bytes,
            "discarded": discarded_bytes,
        },
        "verify_md5": not cfg.no_check,
        "max_redownloads": cfg.max_redownloads,
        "net_retries": cfg.net_retries,
        "flat": cfg.flat,
    }

    manifest = {
        "summary": summary,
        "results": results,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nSummary:")
    print(
        f"  OK: {summary['counts']['ok']} | SKIP-EXISTS: {summary['counts']['skip_exists']} | "
        f"BAD-CHECKSUM: {summary['counts']['bad_checksum']} | FAILED: {summary['counts']['failed']}"
    )
    print(
        f"  Downloaded: {summary['bytes']['downloaded'] / 1024 / 1024:.1f} MB | "
        f"Reused: {summary['bytes']['reused'] / 1024 / 1024:.1f} MB | "
        f"Discarded: {summary['bytes']['discarded'] / 1024 / 1024:.1f} MB"
    )
    print(f"\nWrote manifest to {outdir / 'manifest.json'}")

    return (
        0
        if (summary["counts"]["bad_checksum"] == 0 and summary["counts"]["failed"] == 0)
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
