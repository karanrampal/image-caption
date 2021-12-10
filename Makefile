VENV := ./.venv

install: requirements.txt
	python -m venv $(VENV)
	source $(VENV)/Scripts/activate
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -r requirements.txt

test:
	python -m pytest -vv --cov

format:
	black src
	isort src
	mypy src

lint:
	pylint -j 6 src

clean:
	rm -rf __pycache__ .coverage .mypy_cache *.log

all: install lint test

.PHONY: lint format clean all