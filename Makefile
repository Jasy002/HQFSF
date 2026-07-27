# ==========================================================
# HQFSF Makefile
# ==========================================================

.PHONY: install dev run train evaluate benchmark test \
        format lint clean docker-build docker-run docker-stop

# ----------------------------------------------------------
# Installation
# ----------------------------------------------------------

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

# ----------------------------------------------------------
# Run Project
# ----------------------------------------------------------

run:
	python run.py

train:
	python docs/train.py

evaluate:
	python docs/evaluate.py

benchmark:
	python docs/benchmark.py

# ----------------------------------------------------------
# Testing
# ----------------------------------------------------------

test:
	pytest

# ----------------------------------------------------------
# Formatting & Linting
# ----------------------------------------------------------

format:
	black .
	isort .

lint:
	flake8 .
	pylint .

# ----------------------------------------------------------
# Docker
# ----------------------------------------------------------

docker-build:
	docker compose build

docker-run:
	docker compose up

docker-stop:
	docker compose down

# ----------------------------------------------------------
# Cleanup
# ----------------------------------------------------------

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -f .coverage
	find . -name "*.pyc" -delete