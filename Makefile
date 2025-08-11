VENV ?= .venv
PYTHON ?= python3
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: venv install train evaluate predict

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PIP) install -U pip setuptools wheel
	$(VENV_PIP) install -r requirements.txt

train:
	$(VENV_PY) -m ml_project.src.train --dataset iris --model logistic --output-dir artifacts

evaluate:
	$(VENV_PY) -m ml_project.src.evaluate --artifacts-dir artifacts

predict:
	$(VENV_PY) -m ml_project.src.predict --artifacts-dir artifacts