#!/bin/bash
# Comprehensive environment setup script for the data preprocessing pipeline

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Exit immediately if a command exits with non-zero status
set -e

echo -e "${YELLOW}=== Starting Environment Setup ===${NC}"

# Check for Python installation
echo -e "${YELLOW}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.9+ first.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo -e "${RED}Python 3.9+ is required. Found Python ${PYTHON_VERSION}.${NC}"
    exit 1
fi
echo -e "${GREEN}Found Python ${PYTHON_VERSION}${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated.${NC}"

# Upgrade pip and setuptools
echo -e "${YELLOW}Upgrading pip and setuptools...${NC}"
pip install --upgrade pip setuptools wheel

# Install base requirements
echo -e "${YELLOW}Installing base requirements...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Base requirements installed.${NC}"
else
    echo -e "${RED}requirements.txt not found.${NC}"
    exit 1
fi

# Optional: Install development requirements
if [ "$1" = "dev" ]; then
    echo -e "${YELLOW}Installing development requirements...${NC}"
    if [ -f "infrastructure/requirements/streamlit.txt" ]; then
        pip install -r infrastructure/requirements/streamlit.txt
        echo -e "${GREEN}Development requirements installed.${NC}"
    else
        echo -e "${RED}Development requirements file not found.${NC}"
    fi
    
    # Install in editable mode
    echo -e "${YELLOW}Installing package in editable mode...${NC}"
    pip install -e .
fi

# Set up environment variables
echo -e "${YELLOW}Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}Created .env file from .env.example${NC}"
        echo -e "${YELLOW}Please review and edit the .env file as needed.${NC}"
    else
        echo -e "${YELLOW}No .env.example file found. Creating empty .env file.${NC}"
        touch .env
    fi
else
    echo -e "${YELLOW}.env file already exists.${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/raw data/processed models logs
touch data/raw/.keep data/processed/.keep
echo -e "${GREEN}Directory structure created.${NC}"

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "import pandas; print(f'{GREEN}Successfully imported pandas {pandas.__version__}{NC}')" || \
    echo -e "${RED}Failed to verify installation${NC}"

echo -e "${GREEN}=== Environment Setup Complete ===${NC}"
echo -e "To activate the virtual environment, run: ${YELLOW}source venv/bin/activate${NC}"