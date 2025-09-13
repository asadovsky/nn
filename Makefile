SHELL := /bin/bash -euo pipefail

.DELETE_ON_ERROR:

.PHONY: test
test:
	python -m unittest discover . '*_test.py'

########################################
# Format and lint

.PHONY: fmt
fmt:
	@scm-format

.PHONY: lint
lint:
	@scm-lint

.PHONY: lint-all
lint-all:
	@scm-lint -a
