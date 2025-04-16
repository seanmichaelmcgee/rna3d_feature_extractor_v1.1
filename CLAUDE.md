# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Setup environment: `./setup.sh` or `mamba env create -f environment.yml`
- Activate environment: `mamba activate rna3d-core`
- Run single test: `python -m unittest tests.analysis.test_thermodynamic_analysis`
- Process features: `python scripts/run_feature_extraction_single.sh target_id`

## Code Style Guidelines
- **Imports**: System → third-party → local modules
- **Naming**: snake_case for functions/variables, UPPERCASE for constants
- **Documentation**: Google-style docstrings with Args/Returns sections
- **Error Handling**: Return None for failures, include thorough error checking
- **Types**: No strict typing requirements, use descriptive variable names
- **Structure**: Follow modular design with standardized feature naming: `feature_type.feature_name`
- **Serialization**: Save features in NPZ format with clear naming conventions
- **Performance**: Optimize for memory usage with RNA sequences up to 3,000 nucleotides
- **Dependencies**: Use conditional imports with fallbacks for optional packages

## RNA Tool-Specific Guidelines
- Include validation for thermodynamic consistency
- Ensure proper error handling for ViennaRNA integration
- Create visualizations for debugging RNA structures when appropriate

## Important Files (DO NOT DELETE)
- `notebooks/test_features_extraction.ipynb` - Required for test feature extraction
- `notebooks/train_features_extraction.ipynb` - Required for training feature extraction
- `notebooks/validation_features_extraction.ipynb` - Required for validation feature extraction
- Markdown versions of these notebooks are found in the same directory with _ntbk.py.md extension