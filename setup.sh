#!/bin/bash
# Setup script for RNA 3D Explorer Core

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set error handling
set -e

# Function to verify critical packages are installed correctly
verify_environment() {
    echo -e "${BLUE}Verifying environment...${NC}"
    
    # Attempt to activate the environment
    eval "$(mamba shell.bash hook)" 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null
    
    if ! mamba activate rna3d-core 2>/dev/null && ! conda activate rna3d-core 2>/dev/null; then
        echo -e "${RED}Failed to activate rna3d-core environment${NC}"
        return 1
    fi
    
    # Check if ViennaRNA is available
    if ! python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')" &>/dev/null; then
        echo -e "${RED}ViennaRNA package not properly installed${NC}"
        return 1
    fi
    
    # Check other critical packages
    if ! python -c "import numpy, pandas, matplotlib, torch" &>/dev/null; then
        echo -e "${RED}Core dependencies not properly installed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Environment verification successful${NC}"
    return 0
}

echo -e "${YELLOW}Setting up RNA 3D Explorer Core...${NC}"

# Check if environment.yml exists
if [ ! -f environment.yml ]; then
    echo -e "${RED}Error: environment.yml not found${NC}"
    echo -e "Please ensure you're running this script from the project root directory"
    exit 1
fi

# Make scripts executable
echo -e "${GREEN}Making scripts executable...${NC}"
if [ -f src/core/rna_vis.py ]; then
    chmod +x src/core/rna_vis.py
fi
if [ -f src/core/rna_3d_vis.py ]; then
    chmod +x src/core/rna_3d_vis.py
fi

# Make feature extraction scripts executable
for script in scripts/*.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
    fi
done

# Create mamba environment
echo -e "${GREEN}Setting up environment...${NC}"

# Check for existing environment and ask to update if it exists
ENV_EXISTS=0
if mamba env list | grep -q "rna3d-core"; then
    ENV_EXISTS=1
    echo -e "${YELLOW}The rna3d-core environment already exists.${NC}"
    read -p "Would you like to update it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Skipping environment update.${NC}"
    else
        ENV_EXISTS=2  # Mark for update
    fi
fi

# Create or update the environment
if command -v mamba &> /dev/null; then
    # Mamba is available
    if [ $ENV_EXISTS -eq 0 ]; then
        echo -e "${YELLOW}Creating new mamba environment from environment.yml...${NC}"
        if ! mamba env create -f environment.yml; then
            echo -e "${RED}Failed to create environment${NC}"
            echo -e "${YELLOW}Troubleshooting:${NC}"
            echo -e "1. Check if you have sufficient permissions"
            echo -e "2. Make sure you have enough disk space"
            echo -e "3. Try running: mamba clean --all && mamba env create -f environment.yml"
            exit 1
        fi
    elif [ $ENV_EXISTS -eq 2 ]; then
        echo -e "${YELLOW}Updating existing environment...${NC}"
        if ! mamba env update -f environment.yml; then
            echo -e "${RED}Failed to update environment${NC}"
            echo -e "${YELLOW}You may need to remove and recreate the environment:${NC}"
            echo -e "mamba env remove -n rna3d-core"
            echo -e "mamba env create -f environment.yml"
            exit 1
        fi
    fi
    echo -e "${GREEN}Environment operations completed!${NC}"
elif command -v conda &> /dev/null; then
    # Fall back to conda if mamba isn't available
    echo -e "${YELLOW}Mamba not found, falling back to conda (this may be slower)...${NC}"
    if [ $ENV_EXISTS -eq 0 ]; then
        echo -e "${YELLOW}Creating conda environment from environment.yml...${NC}"
        if ! conda env create -f environment.yml; then
            echo -e "${RED}Failed to create environment with conda${NC}"
            echo -e "${YELLOW}We recommend installing mamba for better dependency resolution:${NC}"
            echo -e "conda install mamba -n base -c conda-forge"
            exit 1
        fi
    elif [ $ENV_EXISTS -eq 2 ]; then
        echo -e "${YELLOW}Updating existing environment with conda...${NC}"
        if ! conda env update -f environment.yml; then
            echo -e "${RED}Failed to update environment with conda${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}Environment created with conda!${NC}"
    echo -e "${YELLOW}Note: For faster environment creation in the future, consider installing mamba:${NC}"
    echo -e "conda install mamba -n base -c conda-forge"
else
    echo -e "${RED}Neither mamba nor conda found.${NC}"
    echo -e "${YELLOW}Please install mamba or conda first:${NC}"
    echo -e "1. Download and install Mambaforge from: https://github.com/conda-forge/miniforge#mambaforge"
    echo -e "2. Run this script again"
    exit 1
fi

# Verify the environment if it was created or updated
if [ $ENV_EXISTS -ne 1 ]; then  # if we created or updated the environment
    if ! verify_environment; then
        echo -e "${RED}Environment verification failed.${NC}"
        echo -e "${YELLOW}This could be due to:${NC}"
        echo -e "1. Package conflicts in the environment.yml"
        echo -e "2. Failed installation of compiled dependencies (especially ViennaRNA)"
        echo -e "3. Incompatibility with your system"
        echo -e "\n${YELLOW}You can try:${NC}"
        echo -e "1. Removing the environment: mamba env remove -n rna3d-core"
        echo -e "2. Recreating with: mamba env create -f environment.yml"
        echo -e "3. If problems persist, check system requirements or contact support"
        exit 1
    fi
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
echo -e "1. Activate the environment: ${BLUE}mamba activate rna3d-core${NC}"
echo -e "2. Run a feature extraction script: ${BLUE}./scripts/run_feature_extraction_single.sh <target_id>${NC}"
echo -e "3. Or explore with Jupyter notebooks: ${BLUE}jupyter notebook notebooks/${NC}"
echo -e "4. Read docs/getting_started.md for more information"
