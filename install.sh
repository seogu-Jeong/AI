#!/bin/bash

# StockSense AI Installation & Runner Script
# This script automates the setup of the Python environment and launches the app.

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}          StockSense AI Setup & Launch            ${NC}"
echo -e "${BLUE}==================================================${NC}"

# 1. Check Python version
echo -e "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)

if [[ -z "$PYTHON_VERSION" ]]; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please download and install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

# Compare version (requires 3.10 or higher)
REQUIRED_VERSION="3.10"
if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}Error: StockSense AI requires Python 3.10 or higher.${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected.${NC}"

# 2. Virtual Environment Setup
VENV_PATH="./venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "Step 2: Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created.${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists.${NC}"
fi

# 3. Activate Environment
echo -e "Step 3: Activating environment and updating pip..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet

# 4. Install Dependencies
REQ_FILE="중간고사 과제/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo -e "Step 4: Installing dependencies (this may take a minute)..."
    pip install -r "$REQ_FILE"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install dependencies.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dependencies installed successfully.${NC}"
else
    echo -e "${RED}Error: requirements.txt not found in '중간고사 과제' directory.${NC}"
    exit 1
fi

# 5. Launch App
echo -e "${BLUE}Step 5: Launching StockSense AI...${NC}"
python3 "중간고사 과제/main.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}Application exited with an error.${NC}"
    exit 1
fi

echo -e "${GREEN}Application closed.${NC}"
