# Streamlined Docker Testing Plan for RNA Feature Extraction

## Overview

This streamlined Docker testing plan focuses on creating a reliable containerized environment for RNA feature extraction that can be used for development and potentially for scaling beyond Kaggle. While Docker isn't directly used in Kaggle, it provides a controlled environment for testing and can serve as foundation for future scaling.

## 1. Simplified Dockerfile

```dockerfile
FROM mambaorg/micromamba:latest

# Set working directory
WORKDIR /app

# Copy environment file for better caching
COPY environment.yml .

# Create environment using mamba (faster than conda)
RUN micromamba env create -f environment.yml && \
    micromamba clean --all --yes

# Set path to include the environment
ENV PATH /opt/conda/envs/rna3d-core/bin:$PATH

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh scripts/*.py

# Create volume mount points for data
VOLUME ["/app/data/raw", "/app/data/processed", "/app/logs"]

# Set default parameters
ENV JOBS=1
ENV PF_SCALE=1.5

# Set entrypoint to run the feature extraction script
ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n rna3d-core ./scripts/run_feature_extraction_single.sh ${JOBS} ${PF_SCALE}"]
```

## 2. Essential Docker Tests

### Basic Environment Validation

Test that the container can be built and has all required dependencies:

```bash
# Build the Docker image
docker build -t rna3d-extractor .

# Verify environment
docker run --rm -it --entrypoint /bin/bash rna3d-extractor -c \
  "micromamba run -n rna3d-core python -c \"import RNA; print(f'ViennaRNA version: {RNA.__version__}')\""
```

### Core Functionality Tests

Test that the container can process a simple RNA sequence:

```bash
# Create directories for test data
mkdir -p test_data/mini
mkdir -p output_test

# Copy a small test file to test_data/mini
cp data/raw/sample_small.fasta test_data/mini/

# Run the container with the test data
docker run --rm \
  -v $(pwd)/test_data/mini:/app/data/raw \
  -v $(pwd)/output_test:/app/data/processed \
  rna3d-extractor
  
# Verify output files were created
ls -la output_test
```

### Memory and Performance Test

Test resource usage with different parameter settings:

```bash
# Monitor resource usage during processing
docker run --rm \
  -v $(pwd)/test_data/mini:/app/data/raw \
  -v $(pwd)/output_test:/app/data/processed \
  --memory=4g --cpus=2 \
  rna3d-extractor

# In another terminal while processing
docker stats
```

## 3. Development Workflow Testing

### Interactive Development in Container

Test the container for interactive development:

```bash
# Run container interactively 
docker run --rm -it \
  -v $(pwd):/app \
  -v $(pwd)/test_data/mini:/app/data/raw \
  -v $(pwd)/output_test:/app/data/processed \
  --entrypoint /bin/bash \
  rna3d-extractor

# Inside container
micromamba run -n rna3d-core python -m unittest tests.analysis.test_thermodynamic_analysis
```

### Local vs. Docker Results Comparison

Verify that results are consistent between local and Docker environments:

```bash
# Run locally
mamba activate rna3d-core
python scripts/run_feature_extraction_single.sh 1 1.5

# Run in Docker
docker run --rm \
  -v $(pwd)/test_data/mini:/app/data/raw \
  -v $(pwd)/output_test:/app/data/processed \
  rna3d-extractor

# Compare outputs
python scripts/compare_outputs.py data/processed/features_test output_test
```

## 4. Potential Scaling Tests

While not immediately needed for Kaggle, these tests prepare for future scaling:

### Batch Processing

```bash
# Test batch processing with multiple sequences
docker run --rm \
  -v $(pwd)/test_data/batch:/app/data/raw \
  -v $(pwd)/output_test:/app/data/processed \
  -e JOBS=4 \
  --entrypoint /bin/bash \
  -c "micromamba run -n rna3d-core python src/data/batch_feature_runner.py --csv data/raw/sequences.csv --limit 10" \
  rna3d-extractor
```

### Resource Constraints Testing

```bash
# Test with limited memory and CPU
docker run --rm \
  -v $(pwd)/test_data/batch:/app/data/raw \
  -v $(pwd)/output_test:/app/data/processed \
  --memory=2g --cpus=1 \
  -e JOBS=2 \
  rna3d-extractor
```

## 5. Practical Docker Usage Guide

### Build the Container

```bash
# Basic build
docker build -t rna3d-extractor .

# Build with no cache for clean install
docker build --no-cache -t rna3d-extractor .
```

### Run the Container

```bash
# Run with default parameters
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  rna3d-extractor

# Run with custom parameters
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  -e JOBS=4 -e PF_SCALE=2.0 \
  rna3d-extractor
```

### Interactive Mode

```bash
# Start interactive session
docker run --rm -it \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  --entrypoint /bin/bash \
  rna3d-extractor

# Inside container, run commands with the environment activated
micromamba run -n rna3d-core python scripts/example.py
```

## 6. Common Issues and Solutions

### Memory Issues

Problem: Container runs out of memory with large sequences

Solution:
```bash
# Increase container memory allocation
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  --memory=8g \
  rna3d-extractor
```

### Permission Issues

Problem: Unable to write to mounted volumes

Solution
```bash
# Check directory permissions
ls -la data/raw data/processed

# Ensure directories have proper permissions
chmod 777 data/raw data/processed
```

### Dependency Issues

Problem: Missing dependencies or conflicts

Solution:
```bash
# Check environment inside container
docker run --rm -it --entrypoint /bin/bash rna3d-extractor -c \
  "micromamba run -n rna3d-core pip list"

# Update environment.yml and rebuild
docker build --no-cache -t rna3d-extractor .
```

## 7. Integration with Development Workflow

For most effective development:

1. Use local environment for rapid iteration and testing
2. Use Docker to verify changes in a clean environment
3. Keep Docker testing focused on critical paths for efficiency
4. Version Docker images alongside code changes


# 8.  Create a simple script to verify feature compatibility
cat > verify_features.py << EOF
#!/usr/bin/env python3
import os
import sys
import numpy as np

def verify_feature_structure(features_dir):
    """Verify features match expected structure for data loader."""
    # Check directory structure
    required_dirs = ['dihedral_features', 'thermo_features', 'mi_features']
    for d in required_dirs:
        if not os.path.isdir(os.path.join(features_dir, d)):
            print(f"ERROR: Required directory {d} not found")
            return False
    
    # Find a sample target
    sample_targets = []
    for file in os.listdir(os.path.join(features_dir, 'thermo_features')):
        if file.endswith('_thermo_features.npz'):
            sample_targets.append(file.replace('_thermo_features.npz', ''))
    
    if not sample_targets:
        print("ERROR: No feature files found to test")
        return False
    
    target_id = sample_targets[0]
    print(f"Verifying features for {target_id}")
    
    # Check file naming and feature structure
    thermo_path = os.path.join(features_dir, 'thermo_features', f"{target_id}_thermo_features.npz")
    dihedral_path = os.path.join(features_dir, 'dihedral_features', f"{target_id}_dihedral_features.npz")
    mi_path = os.path.join(features_dir, 'mi_features', f"{target_id}_features.npz")
    
    # Verify thermodynamic features
    if os.path.exists(thermo_path):
        with np.load(thermo_path) as data:
            # Check for required keys
            if not ('pairing_probs' in data or 'base_pair_probs' in data):
                print("ERROR: Missing pairing_probs/base_pair_probs in thermo features")
                return False
            if not ('positional_entropy' in data or 'position_entropy' in data):
                print("WARNING: Missing positional_entropy in thermo features")
            
            # Check shapes
            key = 'pairing_probs' if 'pairing_probs' in data else 'base_pair_probs'
            shape = data[key].shape
            if len(shape) != 2 or shape[0] != shape[1]:
                print(f"ERROR: Invalid shape for {key}: {shape}")
                return False
            
            print(f"Thermo features: OK, sequence length = {shape[0]}")
    else:
        print(f"WARNING: No thermo features found for {target_id}")
    
    # Verify dihedral features
    if os.path.exists(dihedral_path):
        with np.load(dihedral_path) as data:
            if 'features' not in data:
                print("ERROR: Missing 'features' key in dihedral features")
                return False
            
            shape = data['features'].shape
            if len(shape) != 2 or shape[1] != 4:
                print(f"ERROR: Invalid shape for dihedral features: {shape}")
                return False
            
            print(f"Dihedral features: OK, shape = {shape}")
    
    # Verify MI features
    if os.path.exists(mi_path):
        with np.load(mi_path) as data:
            if 'coupling_matrix' not in data and 'scores' not in data:
                print("ERROR: Missing coupling_matrix/scores in MI features")
                return False
            
            key = 'coupling_matrix' if 'coupling_matrix' in data else 'scores'
            shape = data[key].shape
            if len(shape) != 2 or shape[0] != shape[1]:
                print(f"ERROR: Invalid shape for {key}: {shape}")
                return False
            
            print(f"MI features: OK, shape = {shape}")
    
    print("All features appear compatible with data loader requirements")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_features.py <features_dir>")
        sys.exit(1)
    
    features_dir = sys.argv[1]
    if not verify_feature_structure(features_dir):
        sys.exit(1)
EOF

# Run verification in Docker
docker run --rm \
  -v $(pwd)/data/processed:/app/data/processed \
  --entrypoint /bin/bash \
  -c "micromamba run -n rna3d-core python verify_features.py /app/data/processed" \
  rna3d-extractor

## Conclusion

This streamlined Docker testing approach provides the essential verification needed for the RNA feature extraction pipeline while setting the foundation for future scaling. By focusing on core functionality, resource usage, and practical development workflows, it enables efficient development while ensuring compatibility with both local and containerized environments.

