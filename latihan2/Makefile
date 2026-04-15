PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help setup eda features train eval nlp serve app test lint report clean

help:
	@echo "Targets: setup | eda | features | train | eval | nlp | serve | app | test | lint | report"

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -r src/serve/requirements-serve.txt

# Fase 1-2
eda:
	$(PY) scripts/run_eda.py
	$(PY) scripts/build_eda_notebook.py

features:
	$(PY) scripts/run_feature_engineering.py || true
	$(PY) scripts/build_feature_notebook.py || true

# Fase 3-4
train:
	$(PY) scripts/run_classical_bench.py
	$(PY) scripts/run_dl_bench.py

# Fase 5
nlp:
	$(PY) scripts/run_nlp.py

# Fase 6
eval:
	$(PY) scripts/run_eval_comprehensive.py
	$(PY) scripts/build_eval_notebook.py

# Fase 7
serve:
	PYTHONPATH=. $(PY) -m uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload

# Fase 8
app:
	$(PY) -m streamlit run app/streamlit_app.py --server.port 8501

test:
	PYTHONPATH=. $(PY) -m pytest tests/ -q

lint:
	ruff check src tests scripts

# Fase 9 - render portfolio report
report:
	jupyter nbconvert --to pdf notebooks/06_Evaluation_Report.ipynb \
		--output-dir reports --output portfolio_report

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
