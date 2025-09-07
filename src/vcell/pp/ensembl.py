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
                    self.log_stats_if_needed()
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
    db_fpath = os.path.join(os.path.expandvars(dir), "reports.sqlite")
    db = sqlite3.connect(db_fpath, autocommit=True)

    with open(schema_fpath) as fd:
        schema = fd.read()
    db.executescript(schema)
    db.autocommit = False

    return db


def canonicalize(
    h5: pathlib.Path, dump_to: pathlib.Path, mod: str = "", gene_id_col: str = ""
):
    import json

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
        print("No gene_id_col specified. Available columns in adata.var:")
        print()

        # Get up to 3 example values for each column
        n_examples = min(3, len(adata.var))
        for i, col in enumerate(adata.var.columns, 1):
            # Get example values, handling different data types
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
                print(f"  {i}. {col:<30} (examples: {examples_str})")
            except Exception:
                print(f"  {i}. {col:<30} (could not get examples)")

        while True:
            user_input = input(
                "\nEnter column name (or press Enter to skip gene_id mapping): "
            ).strip()

            if not user_input:
                # User wants to skip - confirm this choice
                confirm = (
                    input("Are you sure you want to skip gene_id mapping? (y/n): ")
                    .strip()
                    .lower()
                )
                if confirm == "y":
                    gene_id_col = ""
                    print("Proceeding without gene_id mapping.")
                    break
                else:
                    continue
            elif user_input in adata.var.columns:
                gene_id_col = user_input
                print(f"Using gene_id_col: '{gene_id_col}'")
                break
            else:
                print(f"Invalid column name '{user_input}'. Please try again.")

    db = get_db(dump_to)
    dataset_stmt = ""
    # TODO: insert the dataset right now.

    symbol_stmt = (
        "INSERT INTO symbols(name, dataset_id, included_ensembl_id) VALUES(?, ?, ?)"
    )
    ensembl_stmt = ""
    map_stmt = ""

    # The client object uses threads, but also avoid going over rate limits. Since we will likely be IO-bound, I'm not worried about using a single process. But we can submit many requests all at once, then try to start getting the results.
    with EnsemblQueryPool() as pool:
        futures = [
            pool.submit(f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{index}")
            for index, row in adata.var.iterrows()
        ]
        for fut, (index, row) in zip(
            helpers.progress(futures, desc="ensembl", every=100), adata.var.iterrows()
        ):
            try:
                # TODO: insert the symbol id
                pass
            except sqlite3.Error as err:
                db.rollback()
                print(f"Error writing symbol blah blah for {index}")
                continue

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
                # TODO: insert the ensembl id, update the ensembl map (via inserts)
                pass
            except sqlite3.Error as err:
                db.rollback()
                print(f"Error writing blah blah for {index}")


def cli():
    import tyro

    tyro.extras.subcommand_cli_from_dict({
        "canonicalize": canonicalize,
        "noop": lambda: None,
    })
