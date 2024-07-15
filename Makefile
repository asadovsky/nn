SHELL := /bin/bash -euo pipefail

.DELETE_ON_ERROR:

.PHONY: test
test:
	python -m unittest discover . '*_test.py'

########################################
# Format and lint

.PHONY: fmt
fmt:
	@./fmt_or_lint.sh -f

.PHONY: lint
lint:
	@./fmt_or_lint.sh

.PHONY: lint-all
lint-all:
	@./fmt_or_lint.sh -a
