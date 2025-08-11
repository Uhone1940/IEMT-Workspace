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

.PHONY: elec_synth elec_train elec_evaluate elec_predict

elec_synth:
	PATH=$$HOME/.local/bin:$$PATH python3 -m ml_project.src.electricity_synth --rows 2000 --output data/electricity_synth.csv

elec_train:
	PATH=$$HOME/.local/bin:$$PATH python3 -m ml_project.src.electricity_train --csv data/electricity_synth.csv --output-dir artifacts_electricity

elec_evaluate:
	PATH=$$HOME/.local/bin:$$PATH python3 -m ml_project.src.electricity_evaluate --artifacts-dir artifacts_electricity

elec_predict:
	PATH=$$HOME/.local/bin:$$PATH python3 -m ml_project.src.electricity_predict --artifacts-dir artifacts_electricity --from-csv data/electricity_synth.csv --head 5