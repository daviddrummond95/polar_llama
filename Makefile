SHELL=/bin/bash

.venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit: .venv
	cargo fmt --all && cargo clippy --all-features
	.venv/bin/python -m ruff check . --fix || true
	if [ -d "polar_llama" ]; then \
		.venv/bin/python -m ruff format polar_llama || true; \
	fi
	if [ -d "tests" ]; then \
		.venv/bin/python -m ruff format tests || true; \
	fi
	if [ -d "polar_llama" ] && [ -d "tests" ]; then \
		.venv/bin/mypy --ignore-missing-imports polar_llama tests || true; \
	elif [ -d "polar_llama" ]; then \
		.venv/bin/mypy --ignore-missing-imports polar_llama || true; \
	elif [ -d "tests" ]; then \
		.venv/bin/mypy --ignore-missing-imports tests || true; \
	fi

test: .venv
	.venv/bin/python -m pytest tests

run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py

