#!/bin/bash
# Setup script for RNA 3D Explorer Core

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up RNA 3D Explorer Core...${NC}"

# Make scripts executable
echo -e "${GREEN}Making scripts executable...${NC}"
chmod +x src/core/rna_vis.py
chmod +x src/core/rna_3d_vis.py

# Create mamba environment
echo -e "${GREEN}Creating mamba environment...${NC}"
if command -v mamba &> /dev/null; then
    echo -e "${YELLOW}Creating mamba environment from environment.yml...${NC}"
    mamba env create -f environment.yml
    echo -e "${GREEN}Environment created! Activate with: mamba activate rna3d-core${NC}"
else
    echo -e "${RED}Mamba not found. Please install mamba and run this script again.${NC}"
    echo -e "${YELLOW}Alternatively, install the dependencies manually using pip:${NC}"
    echo -e "pip install numpy pandas matplotlib seaborn jupyter ipywidgets py3Dmol"
fi

# Create Jupyter notebook from Python script
if [ -f notebooks/core_demo.py ] && ! [ -f notebooks/core_demo.ipynb ]; then
    if command -v jupytext &> /dev/null; then
        echo -e "${GREEN}Converting Python script to Jupyter notebook...${NC}"
        jupytext --to notebook notebooks/core_demo.py
    else
        echo -e "${YELLOW}jupytext not installed. To convert the Python script to a notebook:${NC}"
        echo -e "${YELLOW}  pip install jupytext${NC}"
        echo -e "${YELLOW}  jupytext --to notebook notebooks/core_demo.py${NC}"
    fi
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Activate the environment: mamba activate rna3d-core"
echo "2. Run example script: python src/core/rna_vis.py --list"
echo "3. Read docs/getting_started.md for more information"
