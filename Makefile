PYTHON=python3
VENV_DIR="venv"

.PHONY: venv install activate clean

# Create a virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies if venv exists
install: venv
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Activate the virtual environment (for reference, cannot be run from Makefile)
activate:
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)
