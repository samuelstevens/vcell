Module vcell.helpers
====================

Functions
---------

`current_git_commit() ‑> str | None`
:   Best-effort short SHA of the repo containing *this* file.
    
    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.

Classes
-------

`batched_idx(total_size: int, batch_size: int)`
:   Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.
    
    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.
    
    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    
    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices

`progress(it, *, every: int = 10, desc: str = 'progress', total: int = 0)`
:   Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.
    
    Args:
        it: Iterable to wrap.
        every: How many iterations between logging progress.
        desc: What to name the logger.
        total: If non-zero, how long the iterable is.