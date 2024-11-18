# Variables
VENV_NAME=venv
PYTHON=${VENV_NAME}/bin/python3
PIP=${VENV_NAME}/bin/pip3

# Detect OS
ifeq ($(OS),Windows_NT)
    PYTHON=${VENV_NAME}/Scripts/python3
    PIP=${VENV_NAME}/Scripts/pip3
    ACTIVATE=.\\${VENV_NAME}\\Scripts\\activate
else
    ACTIVATE=source ${VENV_NAME}/bin/activate
endif

.PHONY: all venv clean test run activate inspect

# Default target
all: venv

# Create virtual environment (only needed once)
venv:
	test -d ${VENV_NAME} || python3 -m venv ${VENV_NAME}
	${PIP} install --upgrade pip
	${PIP} install requests pytest black fastapi uvicorn python-dotenv pytest pytest-mock openai


# Show activation command
activate:
	@echo "To activate the virtual environment, run:"
	@echo "${ACTIVATE}"

# Clean up everything (including venv)
clean: clear-db
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf ${VENV_NAME}

run:
	${PYTHON} -m src

# Run tests
test:
	${PYTHON} -m pytest tests/
