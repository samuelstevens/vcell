docs: fmt
    rm -rf docs/api
    mkdir -p docs/api
    uv run pdoc3 --force --output-dir docs/api --config latex_math=True src/vcell
    yek src/vcell *.py *.md > docs/api/llms.txt || true

lint: fmt
    uv run ruff check --fix .

fmt:
    uv run ruff format --preview .

clean:
    rm -rf .ruff_cache/
