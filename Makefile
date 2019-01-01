SHELL := /bin/bash -euo pipefail

.PHONY: lint
lint:
	pylint *.py
