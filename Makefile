SHELL := /bin/bash -euo pipefail

.PHONY: lint
lint:
	pylint *.py

.PHONY: test
test:
	python params_test.py
