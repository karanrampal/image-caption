VENV := ./.venv

install: requirements.txt
	python -m venv $(VENV)
	source $(VENV)/Scripts/activate
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -r requirements.txt

test:
	pytest -vv --cov

format:
	black src tests
	isort src tests
	mypy src tests

lint:
	pylint -j 6 src tests

clean:
	rm -rf __pycache__ .coverage .mypy_cache .pytest_cache *.log

all: install lint test

.PHONY: lint format clean all