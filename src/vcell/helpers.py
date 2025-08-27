# src/vcell/helpers.py
import collections.abc
import dataclasses
import logging
import pathlib
import subprocess
import time
import typing as tp

import beartype


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


@beartype.beartype
class batched_idx:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """

    def __init__(self, total_size: int, batch_size: int):
        """
        Args:
            total_size: total number of examples
            batch_size: maximum distance between the generated indices
        """
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size


@beartype.beartype
def current_git_commit() -> str | None:
    """
    Best-effort short SHA of the repo containing *this* file.

    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.
    """
    try:
        # Walk up until we either hit a .git dir or the FS root
        here = pathlib.Path(__file__).resolve()
        for parent in (here, *here.parents):
            if (parent / ".git").exists():
                break
        else:  # no .git found
            return None

        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


@beartype.beartype
def dict_to_dataclass(data: dict, cls: type) -> tp.Any:
    """Recursively convert a dictionary to a dataclass instance."""
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]

        # Handle Optional types
        origin = tp.get_origin(field_type)
        args = tp.get_args(field_type)

        # Handle tuple[str, ...]
        if origin is tuple and args:
            kwargs[field_name] = tuple(value) if isinstance(value, list) else value
        # Handle list[DataclassType]
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            kwargs[field_name] = [dict_to_dataclass(item, args[0]) for item in value]
        # Handle regular dataclass fields
        elif dataclasses.is_dataclass(field_type):
            kwargs[field_name] = dict_to_dataclass(value, field_type)
        # Handle pathlib.Path
        elif field_type is pathlib.Path or (
            origin is tp.Union and pathlib.Path in args
        ):
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


@beartype.beartype
def get_non_default_values(obj: tp.Any, default_obj: tp.Any) -> dict:
    """Recursively find fields that differ from defaults."""
    obj_dict = dataclasses.asdict(obj)
    default_dict = dataclasses.asdict(default_obj)

    diff = {}
    for key, value in obj_dict.items():
        default_value = default_dict.get(key)
        if value != default_value:
            diff[key] = value

    return diff


@beartype.beartype
def merge_configs(base: tp.Any, overrides: dict) -> tp.Any:
    """Recursively merge override values into a base config."""
    if not overrides:
        return base

    base_dict = dataclasses.asdict(base)

    for key, value in overrides.items():
        if key in base_dict:
            # For nested dataclasses, merge recursively
            if isinstance(value, dict) and dataclasses.is_dataclass(getattr(base, key)):
                base_dict[key] = dataclasses.asdict(
                    merge_configs(getattr(base, key), value)
                )
            else:
                base_dict[key] = value

    return dict_to_dataclass(base_dict, type(base))
