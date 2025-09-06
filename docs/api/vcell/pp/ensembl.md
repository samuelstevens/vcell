Module vcell.pp.ensembl
=======================

Classes
-------

`EnsemblQueryPool(max_workers: int = 8, rps: float = 15.0)`
:   Uses a threadpool to send many queries to the Ensembl REST API. Avoids going over the rate limits via a token bucket implementation.
    
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
    
    Thread-pool executor with rate limiting.

    ### Ancestors (in MRO)

    * concurrent.futures._base.Executor

    ### Methods

    `set_rps(self, rps: float) ‑> None`
    :   Manually adjust RPS.

    `shutdown(self, wait: bool = True, cancel_futures: bool = False) ‑> None`
    :   Shutdown the executor.

    `stats(self) ‑> dict[str, typing.Any]`
    :   Return current statistics. Thread-safe.

    `submit(self, url: str) ‑> concurrent.futures._base.Future`
    :   Schedule a request to a https://rest.ensembl.org url; returns a Future. Respects rate limit and max_workers. Update the rate per second based on the headers.
        
        Our request should ask for JSON and supply a useful User-Agent:
        requests.get(
            url, headers={"Content-Type": "application/json", "User-Agent": "vcell"}
        )
        
        The Future, when resolved to a result, should be the JSON output from the request. We should call raise_for_status() and store any exception in the Future's exception field.

`RequestResult(success: bool, data: object = None, error: Exception | None = None, should_retry: bool = False, wait_seconds: float = 0.0)`
:   Result of an HTTP request attempt.

    ### Instance variables

    `data: object`
    :

    `error: Exception | None`
    :

    `should_retry: bool`
    :

    `success: bool`
    :

    `wait_seconds: float`
    :