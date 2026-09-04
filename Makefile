.PHONY: install install-full test lint demo

install:
	python -m pip install -e ".[dev]"

install-full:
	python -m pip install -e ".[tensorflow,vision,dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

demo:
	python -m facenet_student demo --work-dir runs/demo --epochs 8 --overwrite
