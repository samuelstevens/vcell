# Contributing

## TL;DR

Install [uv](https://docs.astral.sh/uv/).
Clone this repository, then from the root directory:

```sh
uv run python  # TODO
```

You also need [yek](https://github.com/bodo-run/yek) for generating docs.

## Coding Style & Conventions

* Don't hard-wrap comments. Only use linebreaks for new paragraphs. Let the editor soft wrap content.
* Use single-backticks for variables. We use Markdown and [pdoc3](https://pdoc3.github.io/pdoc/) for docs rather than ReST and Sphinx.
* File descriptors from `open()` are called `fd`.
* Use types where possible, including `jaxtyping` hints.
* Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
* Variables referring to a filepath should be suffixed with `_fpath`. Directories are `_dpath`.
* Prefer `make` over `build` when naming functions that construct objects, and use `get` when constructing primitives (like string paths or config values).
* Only use `setup` for naming functions that don't return anything.

Throughout the code, variables are annotated with shape suffixes, as [recommended by Noam Shazeer](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).

The key for these suffixes:

* b: batch size
* d: model dimension

For example, an batched activation tensor with shape (batch, d_model) is `x_bd`.

## Testing & Linting

`justfile` contains commands for testing and linting.

`just lint` will format and lint.
`just test` will format, lint and test, then report coverage.

## Commit / PR Checklist

1. Run `just test`.
2. Check that there are no regressions. Unless you are certain tests are not needed, the coverage % should either stay the same or increase.
3. Run `just docs`.
4. Fix any missing doc links.
