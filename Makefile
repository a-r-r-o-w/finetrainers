.PHONY: quality style

check_dirs := finetrainers tests examples

quality:
	ruff check $(check_dirs) --exclude examples/_legacy
	ruff format --check $(check_dirs) --exclude examples/_legacy

style:
	ruff check $(check_dirs) --fix --exclude examples/_legacy
	ruff format $(check_dirs) --exclude examples/_legacy