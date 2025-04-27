# Variables
MKDOCS = mkdocs
CONFIG_FILE = /workspaces/finetrainers/mkdocs.yml
CHECK_DIRS = finetrainers tests examples train.py setup.py

# Targets
.PHONY: serve build clean quality style

serve:
	$(MKDOCS) serve -f $(CONFIG_FILE)

build:
	$(MKDOCS) build -f $(CONFIG_FILE)

clean:
	rm -rf site/

quality:
	ruff check $(CHECK_DIRS) --exclude examples/_legacy
	ruff format --check $(CHECK_DIRS) --exclude examples/_legacy

style:
	ruff check $(CHECK_DIRS) --fix --exclude examples/_legacy
	ruff format $(CHECK_DIRS) --exclude examples/_legacy
