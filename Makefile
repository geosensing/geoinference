.PHONY: install lint format typecheck docstrings check test docs build clean

install:        ## Sync all dependency groups into the uv environment
	uv sync --all-groups

lint:           ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:         ## Ruff format (writes changes)
	uv run ruff format .

typecheck:      ## Pyright on the package source
	uv run pyright

docstrings:     ## Pydoclint on the package source
	uv run pydoclint geoinference

check:          ## Everything CI runs
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pydoclint geoinference
	uv run pytest
	uvx preen check --strict

test:           ## Run the test suite
	uv run pytest

docs:           ## Build the HTML documentation
	cd docs && make html

build:          ## Build sdist + wheel
	uv build

clean:          ## Remove build/doc artifacts
	rm -rf dist build docs/_build
