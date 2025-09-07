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
def canonicalize(
    h5: pathlib.Path,
    dump_to: pathlib.Path,
    mod: str = "",
    gene_id_col: str = "",
    mode: str = "",
    n_rows: int = 0,
):
    """Canonicalize gene symbols to Ensembl IDs.

    Args:
        h5: Path to the h5ad or h5mu file
        dump_to: Directory to store the SQLite database
        mod: Modality to use (for h5mu files)
        gene_id_col: Column in adata.var containing gene IDs
        mode: How to handle existing datasets:
            - "update": Add new symbols to existing dataset (default)
            - "replace": Delete and recreate existing dataset
            - "new": Always create a new dataset
            - (empty): Ask user interactively
        n_rows: Number of rows to process (0 for all rows)
    """

    import anndata as ad
    import mudata as md
    import numpy as np

    if h5.suffix == ".h5ad":
        adata = ad.read_h5ad(h5, backed="r")
    elif h5.suffix == ".h5mu":
        mdata = md.read_h5mu(h5, backed="r")

        if mdata.n_mod == 1 and not mod:
            mod = mdata.mod_names[0]
            print(f"Assigning mod='{mod}'.")

        if not mod:
            print(f"Need to pass --mod. Available: {mdata.mod_names}")
            return

        if mod not in mdata.mod:
            print(f"Unknown modality --mod '{mod}'. Available: {mdata.mod_names}")
            return

        adata = mdata.mod[mod]
    else:
        print(f"Unknown file type '{h5.suffix}'")
        return

    # Validate gene_id_col
    if gene_id_col:
        if gene_id_col not in adata.var.columns:
            print(f"Error: gene_id_col '{gene_id_col}' not found in adata.var.")
            print(f"Available columns: {list(adata.var.columns)}")
            return
    else:
        # If gene_id_col is empty, list all columns and let user choose
        print("No gene_id_col specified.")

        # Build display function for columns with examples
        def display_column(col):
            n_examples = min(3, len(adata.var))
            try:
                examples = adata.var[col].iloc[:n_examples].tolist()
                # Format examples nicely, truncating long strings
                formatted_examples = []
                for ex in examples:
                    if ex is None or (isinstance(ex, float) and np.isnan(ex)):
                        formatted_examples.append("NaN")
                    elif isinstance(ex, str) and len(ex) > 30:
                        formatted_examples.append(f"{ex[:27]}...")
                    else:
                        formatted_examples.append(str(ex))
                examples_str = ", ".join(formatted_examples)
                return f"{col:<30} (examples: {examples_str})"
            except Exception:
                return f"{col:<30} (could not get examples)"

        # Prompt user to select a column
        selected_col = tui.prompt_selection_from_list(
            "Available columns in adata.var:",
            list(adata.var.columns),
            display_func=display_column,
            allow_skip=True,
            skip_label="Skip gene_id mapping",
        )

        if selected_col is None:
            # User wants to skip - confirm this choice
            if tui.prompt_yes_no("Are you sure you want to skip gene_id mapping?"):
                gene_id_col = ""
                print("Proceeding without gene_id mapping.")
            else:
                # Recursive call to try again
                return canonicalize(h5, dump_to, mod, "", mode, n_rows)
        else:
            gene_id_col = selected_col
            print(f"Using gene_id_col: '{gene_id_col}'")

    db = get_db(dump_to)

    # Normalize path for consistent comparison
    normalized_path = str(h5.resolve())

    # Check if dataset already exists
    existing = db.execute(
        "SELECT dataset_id, name FROM datasets WHERE name = ? AND (gene_id_col = ? OR (gene_id_col IS NULL AND ? IS NULL))",
        (
            normalized_path,
            gene_id_col if gene_id_col else None,
            gene_id_col if gene_id_col else None,
        ),
    ).fetchone()

    if existing:
        dataset_id = existing[0]
        print(f"\nFound existing dataset (ID: {dataset_id}) for {existing[1]}")

        # Get symbol count for this dataset
        symbol_count = db.execute(
            "SELECT COUNT(*) FROM symbols WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()[0]
        print(f"  Currently contains {symbol_count} symbols")

        # Handle mode selection
        if not mode:
            # Interactive mode selection
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

            mode = tui.prompt_choice(
                "How would you like to proceed?", choices=choices, default="update"
            )
            print(f"Using {mode} mode.")

        # Apply the selected mode
        if mode == "update":
            print(f"Updating existing dataset {dataset_id}")
            # dataset_id already set, nothing more to do
        elif mode == "replace":
            print(f"Replacing dataset {dataset_id}, deleting {symbol_count} symbols...")
            db.execute("DELETE FROM symbols WHERE dataset_id = ?", (dataset_id,))
            db.commit()
            print("Existing symbols deleted.")
            # dataset_id already set, will reuse it
        elif mode == "new":
            print("Creating new dataset entry...")
            # Insert new dataset
            dataset_stmt = "INSERT INTO datasets(name, gene_id_col) VALUES(?, ?)"
            cursor = db.execute(
                dataset_stmt, (normalized_path, gene_id_col if gene_id_col else None)
            )
            dataset_id = cursor.lastrowid
            db.commit()
            print(f"Created new dataset with ID: {dataset_id}")
        else:
            print(f"Invalid mode '{mode}'. Must be 'update', 'replace', or 'new'.")
            return
    else:
        # No existing dataset, create new one
        print(f"\nNo existing dataset found for {normalized_path}")
        print("Creating new dataset entry...")
        dataset_stmt = "INSERT INTO datasets(name, gene_id_col) VALUES(?, ?)"
        cursor = db.execute(
            dataset_stmt, (normalized_path, gene_id_col if gene_id_col else None)
        )
        dataset_id = cursor.lastrowid
        db.commit()
        print(f"Created new dataset with ID: {dataset_id}")

    symbol_stmt = "INSERT OR IGNORE INTO symbols(name, dataset_id, included_ensembl_id) VALUES(?, ?, ?)"
    ensembl_stmt = "INSERT OR IGNORE INTO ensembl_genes(gene_id, version, display_name) VALUES(?, ?, ?)"
    map_stmt = "INSERT OR IGNORE INTO symbol_ensembl_map(symbol_id, ensembl_gene_id, source) VALUES(?, ?, ?)"

    # The client object uses threads, but also avoid going over rate limits. Since we will likely be IO-bound, I'm not worried about using a single process. But we can submit many requests all at once, then try to start getting the results.

    # Limit rows if n_rows is specified
    if n_rows > 0:
        import itertools

        var_iter = itertools.islice(adata.var.iterrows(), n_rows)
        var_list = list(var_iter)
        print(f"Processing first {n_rows} rows for testing")
    else:
        var_list = list(adata.var.iterrows())
        print(f"Processing all {len(var_list)} rows")

    # First pass: identify which symbols need API calls
    symbols_needing_api = []
    symbol_to_row = {}
    symbols_with_mappings = 0

    for index, row in var_list:
        # Check if symbol already exists for this dataset
        existing_symbol = db.execute(
            "SELECT symbol_id FROM symbols WHERE name = ? AND dataset_id = ?",
            (index, dataset_id),
        ).fetchone()

        if existing_symbol:
            symbol_id = existing_symbol[0]
            # Check if this symbol already has Ensembl mappings
            existing_mappings = db.execute(
                "SELECT COUNT(*) FROM symbol_ensembl_map WHERE symbol_id = ?",
                (symbol_id,),
            ).fetchone()[0]

            if existing_mappings > 0:
                symbols_with_mappings += 1
                continue  # Skip API call for this symbol
        else:
            # Insert new symbol
            cursor = db.execute(symbol_stmt, (index, dataset_id, None))
            symbol_id = cursor.lastrowid
            db.commit()

        # This symbol needs an API call
        symbols_needing_api.append((index, row, symbol_id))
        symbol_to_row[index] = (row, symbol_id)

    if symbols_with_mappings > 0:
        print(
            f"Skipping {symbols_with_mappings} symbols that already have Ensembl mappings"
        )

    if not symbols_needing_api:
        print("All symbols already have Ensembl mappings, nothing to query")
        return

    print(f"Querying Ensembl API for {len(symbols_needing_api)} symbols")

    with EnsemblQueryPool() as pool:
        # Submit API requests only for symbols that need them
        futures = {}
        for index, row, symbol_id in symbols_needing_api:
            future = pool.submit(
                f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{index}"
            )
            futures[future] = (index, row, symbol_id)

        # Process results as they complete
        for fut in helpers.progress(
            futures.keys(), desc="ensembl", every=100 if len(futures) > 100 else 10
        ):
            index, row, symbol_id = futures[fut]

            err = fut.exception()
            if err is not None:
                print(f"Failed on {index}: {err}")
                continue

            result = fut.result()
            output_dict = {"ensembl": result, "symbol": index}
            # Only add gene_id if gene_id_col is specified
            if gene_id_col:
                output_dict["gene_id"] = row[gene_id_col]
            try:
                # Process each ensembl result from the API
                for item in result:
                    if item.get("type") == "gene":
                        gene_id = item.get("id", "")
                        # Split gene_id to get base ID and version if present
                        if "." in gene_id:
                            base_id, version = gene_id.rsplit(".", 1)
                        else:
                            base_id, version = gene_id, None

                        # Insert ensembl gene
                        cursor = db.execute(
                            ensembl_stmt,
                            (base_id, version, item.get("description", "")),
                        )

                        # Get the ensembl_id (either newly inserted or existing)
                        ensembl_id = db.execute(
                            "SELECT ensembl_id FROM ensembl_genes WHERE gene_id = ?",
                            (base_id,),
                        ).fetchone()[0]

                        # Create mapping between symbol and ensembl gene
                        db.execute(map_stmt, (symbol_id, ensembl_id, "ensembl_xrefs"))

                        # Update symbol with included_ensembl_id if this is the first/primary match
                        db.execute(
                            "UPDATE symbols SET included_ensembl_id = ? WHERE symbol_id = ? AND included_ensembl_id IS NULL",
                            (ensembl_id, symbol_id),
                        )

                db.commit()
            except sqlite3.Error as err:
                db.rollback()
                print(f"Error writing ensembl data for symbol '{index}': {err}")


def cli():
    import tyro

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    tyro.extras.subcommand_cli_from_dict({
        "canonicalize": canonicalize,
        "noop": lambda: None,
    })
