PYTHON=python3
VENV_DIR=".venv"

.PHONY: venv install activate clean test run

# Create a virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies if venv exists
install: venv
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

test: activate
	$(VENV_DIR)/bin/pip install -r requirements-test.txt
	pytest

# Activate the virtual environment (for reference, cannot be run from Makefile)
activate: install
	@source $(VENV_DIR)/bin/activate

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)

# Run the main.py file
run:
	$(VENV_DIR)/bin/python main.py

load_knowledge_base:
	$(VENV_DIR)/bin/python -m knowledge_base.run_job