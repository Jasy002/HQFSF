install:
	pip install -r requirements.txt

run:
	python run.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

benchmark:
	python scripts/benchmark.py

test:
	pytest

format:
	black .

lint:
	flake8 .

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache