import concurrent.futures
import dataclasses
import logging
import os
import pathlib
import queue
import sqlite3
import threading
import time
import typing as tp

import beartype
import pandas as pd
import requests

from .. import helpers
from ..utils import tui

schema_fpath = pathlib.Path(__file__).parent / "ensembl_schema.sql"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RequestResult:
    """Result of an HTTP request attempt."""

    success: bool
    data: object = None
    error: Exception | None = None
    should_retry: bool = False
    wait_seconds: float = 0.0


@beartype.beartype
class EnsemblQueryPool(concurrent.futures.Executor):
    """Uses a threadpool to send many queries to the Ensembl REST API. Avoids going over the rate limits via a token bucket implementation.

    The rate limit can be adjusted based on the headers in the response.

    From https://github.com/Ensembl/ensembl-rest/wiki/Rate-Limits, headers look like this and inform our rate limits:

    X-RateLimit-Limit: 55000
    X-RateLimit-Reset: 892
    X-RateLimit-Period: 3600
    X-RateLimit-Remaining: 54999

    We might also get a header after maxing out that looks like this:

    Retry-After: 40.0
    X-RateLimit-Limit: 55000
    X-RateLimit-Reset: 40
    X-RateLimit-Period: 3600
    X-RateLimit-Remaining: 0

    This means we must wait 40 seconds before sending another request.
    """

    def __init__(self, max_workers: int = 8, rps: float = 15.0):
        """Thread-pool executor with rate limiting."""
        self._max_workers = max_workers
        self._rps = rps
        self._lock = threading.Lock()

        # Set up logger
        self._logger = logging.getLogger(__name__)

        # Token bucket state
        self._tokens = 0
        self._max_tokens = rps
        self._last_refill = time.monotonic()

        # Retry-After handling
        self._retry_after_until = 0.0  # Absolute time when we can resume

        # Queue for pending work
        self._work_queue: queue.Queue[tuple[str, concurrent.futures.Future]] = (
            queue.Queue()
        )

        # Statistics
        self._in_flight = 0
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._retry_after_hits = 0
        self._last_stats_log = 0

        # Shutdown flag
        self._shutdown = False

        # Start worker threads
        self._workers = []
        for _ in range(max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def _update_tokens(self, now: float):
        """Update token bucket. Must be called with _lock held."""
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._rps
        self._tokens = min(self._tokens + tokens_to_add, self._max_tokens)
        self._last_refill = now

    def _acquire_token(self) -> float:
        """Acquire a rate limit token. Returns wait time if needed. Thread-safe."""
        while True:
            with self._lock:
                now = time.monotonic()

                # Check if we're in a Retry-After period
                if now < self._retry_after_until:
                    return self._retry_after_until - now

                # Update tokens
                self._update_tokens(now)

                # Try to consume a token
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return 0.0  # No wait needed

                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._rps

            # Wait without holding lock
            time.sleep(wait_time)

    def _extract_rate_limit_info(
        self, response: requests.Response
    ) -> tuple[float, float]:
        """Extract rate limit info from response. Returns (retry_after_seconds, new_rps)."""
        retry_after = 0.0
        new_rps = self._rps

        # Check for Retry-After header
        if "Retry-After" in response.headers:
            retry_after = float(response.headers["Retry-After"])

        # Adjust RPS based on remaining quota
        if "X-RateLimit-Remaining" in response.headers:
            remaining = int(response.headers["X-RateLimit-Remaining"])
            if remaining < 1000:
                new_rps = max(1.0, new_rps * 0.5)
            elif remaining < 5000:
                new_rps = max(5.0, new_rps * 0.8)

        # Respect absolute limits
        if (
            "X-RateLimit-Limit" in response.headers
            and "X-RateLimit-Period" in response.headers
        ):
            limit = int(response.headers["X-RateLimit-Limit"])
            period = int(response.headers["X-RateLimit-Period"])
            max_rps = (limit / period) * 0.8  # Use 80% of max
            new_rps = min(new_rps, max_rps)

        return retry_after, new_rps

    def _make_request(self, url: str) -> RequestResult:
        """Make HTTP request and return structured result."""
        try:
            response = requests.get(
                url,
                headers={"Content-Type": "application/json", "User-Agent": "vcell"},
                timeout=30,
            )

            # Extract rate limit info (pure function, no side effects)
            retry_after, new_rps = self._extract_rate_limit_info(response)

            # Update state based on response
            with self._lock:
                if retry_after > 0:
                    self._retry_after_until = time.monotonic() + retry_after
                    self._tokens = 0
                    self._retry_after_hits += 1
                    self._logger.warning(
                        f"Hit Retry-After limit, waiting {retry_after:.1f}s. "
                        f"retry_after_hits={self._retry_after_hits}, total_requests={self._total_requests}"
                    )
                self._rps = new_rps
                self._max_tokens = new_rps

            # Check status and return result
            if response.status_code == 429:
                return RequestResult(
                    success=False,
                    error=requests.HTTPError("Rate limited (429)", response=response),
                    should_retry=True,
                    wait_seconds=retry_after,
                )
            elif response.ok:
                return RequestResult(success=True, data=response.json())
            else:
                response.raise_for_status()
                return RequestResult(success=False)  # Shouldn't reach here

        except requests.RequestException as e:
            return RequestResult(success=False, error=e, should_retry=False)
        except Exception as e:
            return RequestResult(success=False, error=e, should_retry=False)

    def _log_stats_if_needed(self) -> None:
        """Log stats every 100 requests. Must be called with _lock held."""
        if self._total_requests > 0 and self._total_requests % 100 == 0:
            if self._total_requests != self._last_stats_log:
                self._last_stats_log = self._total_requests
                self._logger.info(
                    f"total={self._total_requests}, "
                    f"successful={self._successful_requests}, "
                    f"failed={self._failed_requests}, "
                    f"retry_after_hits={self._retry_after_hits}, "
                    f"in_flight={self._in_flight}, "
                    f"queued={self._work_queue.qsize()}, "
                    f"rps={self._rps:.1f}, "
                    f"tokens={self._tokens:.1f}"
                )

    def _process_request(self, url: str) -> RequestResult:
        """Process a single request with retries. Thread-safe."""
        max_retries = 3
        result = RequestResult(success=False, error=Exception("No attempts made"))

        for attempt in range(max_retries):
            # Wait for rate limit token
            wait_time = self._acquire_token()
            while wait_time > 0:
                time.sleep(wait_time)
                # Re-acquire after waiting
                wait_time = self._acquire_token()

            # Make the request
            result = self._make_request(url)

            # Handle result
            if result.success:
                return result
            elif result.should_retry and attempt < max_retries - 1:
                if result.wait_seconds > 0:
                    time.sleep(result.wait_seconds)
                continue
            else:
                return result

        # Shouldn't reach here, but return last result
        return result

    def _worker_loop(self) -> None:
        """Worker thread that processes queued requests."""
        while not self._shutdown:
            try:
                # Get work from queue
                url, future = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Skip cancelled futures
            if future.cancelled():
                self._work_queue.task_done()
                continue

            # Track in-flight request
            with self._lock:
                self._in_flight += 1

            try:
                # Process the request with retries
                result = self._process_request(url)

                # Update statistics and set future result
                with self._lock:
                    if result.success:
                        self._successful_requests += 1
                        future.set_result(result.data)
                    else:
                        self._failed_requests += 1
                        future.set_exception(result.error or Exception("Unknown error"))
            except Exception as e:
                # Catch any unexpected errors
                with self._lock:
                    self._failed_requests += 1
                future.set_exception(e)
            finally:
                # Clean up
                with self._lock:
                    self._in_flight -= 1
                    self._total_requests += 1
                    self._log_stats_if_needed()
                self._work_queue.task_done()

    def submit(self, url: str) -> concurrent.futures.Future:
        """Schedule a request to a https://rest.ensembl.org url; returns a Future. Respects rate limit and max_workers. Update the rate per second based on the headers.

        Our request should ask for JSON and supply a useful User-Agent:
        requests.get(
            url, headers={"Content-Type": "application/json", "User-Agent": "vcell"}
        )

        The Future, when resolved to a result, should be the JSON output from the request. We should call raise_for_status() and store any exception in the Future's exception field.
        """
        if self._shutdown:
            raise RuntimeError("Cannot submit to a shutdown executor")

        future = concurrent.futures.Future()
        self._work_queue.put((url, future))
        return future

    def set_rps(self, rps: float) -> None:
        """Manually adjust RPS."""
        with self._lock:
            self._rps = rps
            self._max_tokens = rps

    def stats(self) -> dict[str, tp.Any]:
        """Return current statistics. Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._update_tokens(now)

            # Calculate next permit time
            if self._tokens >= 1.0:
                next_permit_at = 0.0
            elif now < self._retry_after_until:
                next_permit_at = self._retry_after_until - now
            else:
                next_permit_at = (1.0 - self._tokens) / self._rps

            return {
                "tokens": self._tokens,
                "effective_rps": self._rps,
                "in_flight": self._in_flight,
                "queued": self._work_queue.qsize(),
                "next_permit_at": next_permit_at,
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "retry_after_hits": self._retry_after_hits,
            }

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shutdown the executor."""
        self._shutdown = True

        if cancel_futures:
            # Cancel all pending futures
            while not self._work_queue.empty():
                try:
                    _, future = self._work_queue.get_nowait()
                    future.cancel()
                    self._work_queue.task_done()
                except queue.Empty:
                    break

        if wait:
            # Wait for all workers to finish
            for worker in self._workers:
                worker.join(timeout=5.0)

            # Wait for queue to be processed
            self._work_queue.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


@beartype.beartype
def get_db(dir: str | os.PathLike) -> sqlite3.Connection:
    """Get a connection to the cached ensembl database.

    Args:
        dir: Where to look.

    Returns:
        sqlite3.Connection: A connection to the SQLite database
    """
    os.makedirs(os.path.expandvars(dir), exist_ok=True)
    helpers.warn_if_nfs(dir)
    db_fpath = os.path.join(os.path.expandvars(dir), "ensembl.sqlite")
    db = sqlite3.connect(db_fpath, autocommit=True)

    with open(schema_fpath) as fd:
        schema = fd.read()
    db.executescript(schema)
    db.autocommit = False

    return db


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    """Information about a dataset in the database."""

    dataset_id: int
    """The database ID of this dataset."""

    name: str
    """The file path of the dataset."""

    symbol_count: int
    """Number of symbols in this dataset."""

    is_new: bool
    """Whether this is a newly created dataset."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SymbolInfo:
    """Information about a symbol to process."""

    symbol_id: str
    """The symbol name (gene name)."""

    row: tp.Any
    """The row from adata.var."""

    needs_api_call: bool
    """Whether this symbol needs an API call."""


@beartype.beartype
def load_adata(h5_fpath: pathlib.Path, mod: str = ""):
    """Load AnnData from h5ad or h5mu file.

    Returns None if file type is unsupported or modality is missing.
    """
    import anndata as ad
    import mudata as md

    if h5_fpath.suffix == ".h5ad":
        return ad.read_h5ad(h5_fpath, backed="r")

    if h5_fpath.suffix != ".h5mu":
        print(f"Unknown file type '{h5_fpath.suffix}'")
        return None

    # Handle h5mu files
    mdata = md.read_h5mu(h5_fpath, backed="r")

    # Auto-select if only one modality
    if mdata.n_mod == 1 and not mod:
        mod = mdata.mod_names[0]
        print(f"Assigning mod='{mod}'.")

    if not mod:
        print(f"Need to pass --mod. Available: {mdata.mod_names}")
        return None

    if mod not in mdata.mod:
        print(f"Unknown modality --mod '{mod}'. Available: {mdata.mod_names}")
        return None

    return mdata.mod[mod]


@beartype.beartype
def get_gene_id_column(adata, gene_id_col: str = "") -> str:
    """Get or prompt for gene_id column.

    Returns empty string if user chooses to skip.
    """
    import numpy as np

    # If specified, validate it exists
    if gene_id_col:
        if gene_id_col not in adata.var.columns:
            print(f"Error: gene_id_col '{gene_id_col}' not found in adata.var.")
            print(f"Available columns: {list(adata.var.columns)}")
            return ""  # Invalid column
        return gene_id_col

    # Otherwise prompt user
    print("No gene_id_col specified.")

    def display_column(col):
        """Format column with example values."""
        n_examples = min(3, len(adata.var))
        try:
            examples = adata.var[col].iloc[:n_examples].tolist()
            formatted = []
            for ex in examples:
                if ex is None or (isinstance(ex, float) and np.isnan(ex)):
                    formatted.append("NaN")
                elif isinstance(ex, str) and len(ex) > 30:
                    formatted.append(f"{ex[:27]}...")
                else:
                    formatted.append(str(ex))
            return f"{col:<30} (examples: {', '.join(formatted)})"
        except Exception:
            return f"{col:<30} (could not get examples)"

    selected = tui.prompt_selection_from_list(
        "Available columns in adata.var:",
        list(adata.var.columns),
        display_func=display_column,
        allow_skip=True,
        skip_label="Skip gene_id mapping",
    )

    if selected is None:
        if tui.prompt_yes_no("Are you sure you want to skip gene_id mapping?"):
            print("Proceeding without gene_id mapping.")
            return ""
        # User changed mind, prompt again
        return get_gene_id_column(adata, "")

    print(f"Using gene_id_col: '{selected}'")
    return selected


@beartype.beartype
def prompt_dataset_mode(symbol_count: int) -> str:
    """Prompt user for how to handle existing dataset."""
    choices = [
        tui.Choice(
            key="1",
            value="update",
            label="update",
            description="Add new symbols to existing dataset",
        ),
        tui.Choice(
            key="2",
            value="replace",
            label="replace",
            description="Delete existing dataset and reimport",
            requires_confirmation=True,
            confirmation_prompt=f"Are you sure you want to delete {symbol_count} existing symbols? (y/n): ",
        ),
        tui.Choice(
            key="3",
            value="new",
            label="new",
            description="Create a new dataset entry",
        ),
    ]

    return tui.prompt_choice(
        "How would you like to proceed?", choices=choices, default="update"
    )


@beartype.beartype
def get_or_create_dataset(
    db: sqlite3.Connection, h5_fpath: pathlib.Path, gene_id_col: str, mode: str = ""
) -> DatasetInfo | None:
    """Get or create a dataset in the database.

    Returns None if mode is invalid.
    """
    normalized_path = str(h5_fpath.resolve())

    # Check for existing dataset
    existing = db.execute(
        "SELECT dataset_id, name FROM datasets WHERE name = ? AND (gene_id_col = ? OR (gene_id_col IS NULL AND ? IS NULL))",
        (normalized_path, gene_id_col or None, gene_id_col or None),
    ).fetchone()

    if not existing:
        # No existing dataset - create new one
        print(f"\nNo existing dataset found for {normalized_path}")
        print("Creating new dataset entry...")
        cursor = db.execute(
            "INSERT INTO datasets(name, gene_id_col) VALUES(?, ?)",
            (normalized_path, gene_id_col or None),
        )
        dataset_id = cursor.lastrowid
        db.commit()
        if dataset_id is None:
            raise RuntimeError("Failed to create dataset")
        print(f"Created new dataset with ID: {dataset_id}")
        return DatasetInfo(dataset_id, normalized_path, 0, True)

    # Dataset exists - handle based on mode
    dataset_id = existing[0]
    print(f"\nFound existing dataset (ID: {dataset_id}) for {existing[1]}")

    symbol_count = db.execute(
        "SELECT COUNT(*) FROM dataset_symbols WHERE dataset_id = ?", (dataset_id,)
    ).fetchone()[0]
    print(f"  Currently contains {symbol_count} symbols")

    # Get mode if not specified
    if not mode:
        mode = prompt_dataset_mode(symbol_count)
        print(f"Using {mode} mode.")

    # Apply mode
    if mode == "update":
        print(f"Updating existing dataset {dataset_id}")
        return DatasetInfo(dataset_id, normalized_path, symbol_count, False)

    if mode == "replace":
        print(f"Replacing dataset {dataset_id}, deleting {symbol_count} symbols...")
        db.execute("DELETE FROM dataset_symbols WHERE dataset_id = ?", (dataset_id,))
        db.commit()
        print("Existing symbols deleted.")
        return DatasetInfo(dataset_id, normalized_path, 0, False)

    if mode == "new":
        print("Creating new dataset entry...")
        cursor = db.execute(
            "INSERT INTO datasets(name, gene_id_col) VALUES(?, ?)",
            (normalized_path, gene_id_col or None),
        )
        dataset_id = cursor.lastrowid
        db.commit()
        if dataset_id is None:
            raise RuntimeError("Failed to create dataset")
        print(f"Created new dataset with ID: {dataset_id}")
        return DatasetInfo(dataset_id, normalized_path, 0, True)

    print(f"Invalid mode '{mode}'. Must be 'update', 'replace', or 'new'.")
    return None


@beartype.beartype
def prepare_symbols(
    db: sqlite3.Connection, var_list: list[tuple[str, tp.Any]], dataset_id: int
) -> list[SymbolInfo]:
    """Prepare symbols for processing, determining which need API calls.

    Returns list of SymbolInfo objects that need processing.
    """
    symbols_to_process = []
    symbols_with_mappings = 0

    gene_symbol_stmt = "INSERT OR IGNORE INTO gene_symbols(symbol_id) VALUES(?)"
    dataset_symbol_stmt = "INSERT OR IGNORE INTO dataset_symbols(symbol_id, dataset_id, original_gene_id) VALUES(?, ?, ?)"

    for symbol_id, row in var_list:
        # Check if symbol exists in gene_symbols
        existing = db.execute(
            "SELECT symbol_id FROM gene_symbols WHERE symbol_id = ?",
            (symbol_id,),
        ).fetchone()

        if existing:
            # Symbol exists, check if it has mappings
            mapping_count = db.execute(
                "SELECT COUNT(*) FROM symbol_ensembl_map WHERE symbol_id = ?",
                (symbol_id,),
            ).fetchone()[0]

            if mapping_count > 0:
                symbols_with_mappings += 1
                # Still need to add to dataset_symbols if not already there
                db.execute(dataset_symbol_stmt, (symbol_id, dataset_id, None))
                db.commit()
                continue  # Skip API call for this symbol

            needs_api = True
        else:
            # Create new symbol
            db.execute(gene_symbol_stmt, (symbol_id,))
            db.commit()
            needs_api = True

        # Add to dataset_symbols
        db.execute(dataset_symbol_stmt, (symbol_id, dataset_id, None))
        db.commit()

        symbols_to_process.append(SymbolInfo(symbol_id, row, needs_api))

    if symbols_with_mappings > 0:
        print(
            f"Skipping {symbols_with_mappings} symbols that already have Ensembl mappings"
        )

    return symbols_to_process


@beartype.beartype
def process_ensembl_result(
    db: sqlite3.Connection,
    symbol_info: SymbolInfo,
    api_result: list[dict],
    dataset_id: int,
    gene_id_col: str = "",
) -> None:
    """Process Ensembl API result and update database."""
    ensembl_stmt = (
        "INSERT OR IGNORE INTO ensembl_genes(ensembl_gene_id, name) VALUES(?, ?)"
    )
    map_stmt = "INSERT OR IGNORE INTO symbol_ensembl_map(symbol_id, ensembl_gene_id, source) VALUES(?, ?, ?)"

    for item in api_result:
        if item.get("type") != "gene":
            continue

        gene_id = item.get("id", "")
        if not gene_id:
            continue

        # Store full ID with version suffix
        ensembl_gene_id = gene_id

        # Insert ensembl gene
        db.execute(ensembl_stmt, (ensembl_gene_id, item.get("description", "")))
        db.commit()  # Commit immediately to ensure the gene exists

        # Verify the gene was inserted or already exists
        exists = db.execute(
            "SELECT ensembl_gene_id FROM ensembl_genes WHERE ensembl_gene_id = ?",
            (ensembl_gene_id,),
        ).fetchone()

        if exists:
            # Create mapping
            db.execute(
                map_stmt, (symbol_info.symbol_id, ensembl_gene_id, "ensembl_xrefs")
            )

    # If we have a gene_id column, update original_gene_id in dataset_symbols
    if gene_id_col and hasattr(symbol_info.row, "__getitem__"):
        original_id = symbol_info.row.get(gene_id_col)
        if original_id and not pd.isna(original_id):
            # First ensure the original ID exists in ensembl_genes
            original_id_str = str(original_id)
            db.execute(
                "INSERT OR IGNORE INTO ensembl_genes(ensembl_gene_id, name) VALUES(?, ?)",
                (original_id_str, None),
            )
            # Now safe to update dataset_symbols
            db.execute(
                "UPDATE dataset_symbols SET original_gene_id = ? WHERE symbol_id = ? AND dataset_id = ?",
                (original_id_str, symbol_info.symbol_id, dataset_id),
            )

    db.commit()


@beartype.beartype
def canonicalize(
    h5: pathlib.Path,
    dump_to: pathlib.Path,
    mod: str = "",
    gene_id_col: str = "",
    mode: str = "",
    n_rows: int = 0,
):
    """Canonicalize gene symbols to Ensembl IDs with cleaner structure.

    Args:
        h5: Path to the h5ad or h5mu file
        dump_to: Directory to store the SQLite database
        mod: Modality to use (for h5mu files)
        gene_id_col: Column in adata.var containing gene IDs
        mode: How to handle existing datasets:
            - "update": Add new symbols to existing dataset
            - "replace": Delete and recreate existing dataset
            - "new": Always create a new dataset
            - (empty): Ask user interactively
        n_rows: Number of rows to process (0 for all rows)
    """
    # I ran:
    #
    # uv run ensembl canonicalize --h5 /Volumes/samuel-stevens-2TB/datasets/scperturb/13350497/NadigOConner2024_jurkat.h5ad --dump-to data/cached/ --gene-id-col ensembl_id --mode update
    # uv run ensembl canonicalize --h5 /Volumes/samuel-stevens-2TB/datasets/scperturb/13350497/NadigOConner2024_hepg2.h5ad --dump-to data/cached/ --gene-id-col ensembl_id --mode update
    # uv run ensembl canonicalize --h5 /Volumes/samuel-stevens-2TB/datasets/scperturb/13350497/ReplogleWeissman2022_K562_essential.h5ad --dump-to data/cached/ --gene-id-col ensembl_id --mode update
    # uv run ensembl canonicalize --h5 /Volumes/samuel-stevens-2TB/datasets/scperturb/13350497/ReplogleWeissman2022_K562_gwps.h5ad --dump-to data/cached/ --gene-id-col ensembl_id --mode update
    # uv run ensembl canonicalize --h5 /Volumes/samuel-stevens-2TB/datasets/KOLF_Pan_Genome_Aggregate.h5mu --dump-to data/cached/ --mod rna --gene-id-col gene_ids --mode update
    # uv run ensembl canonicalize --h5 data/inputs/vcc/adata_Training.h5ad --dump-to data/cached/ --gene-id-col gene_ids --mode update

    import itertools

    # Step 1: Load the data
    adata = load_adata(h5, mod)
    if adata is None:
        return

    # Step 2: Get gene_id column
    gene_id_col = get_gene_id_column(adata, gene_id_col)

    # Step 3: Setup database and dataset
    db = get_db(dump_to)
    dataset_info = get_or_create_dataset(db, h5, gene_id_col, mode)
    if dataset_info is None:
        return

    # Step 4: Get list of variables to process
    if n_rows > 0:
        var_iter = itertools.islice(adata.var.iterrows(), n_rows)
        var_list = list(var_iter)
        print(f"Processing first {n_rows} rows for testing")
    else:
        var_list = list(adata.var.iterrows())
        print(f"Processing all {len(var_list)} rows")

    # Step 5: Prepare symbols and identify which need API calls
    symbols = prepare_symbols(db, var_list, dataset_info.dataset_id)

    if not symbols:
        print("All symbols already have Ensembl mappings, nothing to query")
        return

    print(f"Querying Ensembl API for {len(symbols)} symbols")

    # Step 6: Query API and process results
    with EnsemblQueryPool() as pool:
        # Submit all API requests
        futures = {}
        for symbol in symbols:
            if symbol.needs_api_call:
                url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{symbol.symbol_id}"
                future = pool.submit(url)
                futures[future] = symbol

        # Process results
        progress_iter = helpers.progress(
            futures.keys(), desc="ensembl", every=100 if len(futures) > 100 else 10
        )

        for future in progress_iter:
            symbol = futures[future]

            # Check for errors
            err = future.exception()
            if err is not None:
                print(f"Failed on {symbol.symbol_id}: {err}")
                continue

            # Process successful result
            try:
                result = future.result()
                # Build output dict for logging (optional)
                output_dict = {"ensembl": result, "symbol": symbol.symbol_id}
                if gene_id_col and hasattr(symbol.row, "__getitem__"):
                    output_dict["gene_id"] = symbol.row[gene_id_col]

                process_ensembl_result(
                    db, symbol, result, dataset_info.dataset_id, gene_id_col
                )
            except sqlite3.Error as err:
                # Only rollback if we're in a transaction
                try:
                    db.rollback()
                except sqlite3.Error:
                    pass  # Already not in a transaction
                print(f"Database error for symbol '{symbol.symbol_id}': {err}")


def cli():
    import tyro

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    tyro.extras.subcommand_cli_from_dict({
        "canonicalize": canonicalize,
        "noop": lambda: None,
    })
