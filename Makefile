# Variables
MKDOCS = mkdocs
CONFIG_FILE = /workspaces/finetrainers/mkdocs.yml
CHECK_DIRS = finetrainers tests examples train.py setup.py
DOCS_DIR = /workspaces/finetrainers/static_docs/source

# Targets
.PHONY: serve build clean quality style

serve:
	$(MKDOCS) serve -f $(CONFIG_FILE)

build:
	$(MKDOCS) build -f $(CONFIG_FILE) --site-dir $(DOCS_DIR)

clean:
	rm -rf site/

quality:
	ruff check $(CHECK_DIRS) --exclude examples/_legacy
	ruff format --check $(CHECK_DIRS) --exclude examples/_legacy

style:
	ruff check $(CHECK_DIRS) --fix --exclude examples/_legacy
	ruff format $(CHECK_DIRS) --exclude examples/_legacy