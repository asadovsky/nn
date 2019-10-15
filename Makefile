SHELL := /bin/bash -euo pipefail

.PHONY: lint
lint:
	pylint *.py

.PHONY: test
test:
	python intent_prediction_test.py
	python params_test.py
