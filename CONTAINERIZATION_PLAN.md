# RNA 3D Feature Extractor Containerization Plan

This document outlines the containerization strategy for the RNA 3D Feature Extractor, focusing on making it compatible with Kaggle and similar platforms while maintaining local development capabilities.

## 1. Containerization Objectives

- Create a portable, reproducible environment for RNA feature extraction
- Ensure compatibility with Kaggle notebooks and competitions
- Support both interactive notebook and batch processing modes
- Maintain ability to use Mamba/Conda as an alternative local option
- Optimize for performance in resource-constrained environments

## 2. Container Architecture

### 2.1 Base Image

We will use the existing Dockerfile as a starting point, which is based on `mambaorg/micromamba:latest`. This provides several advantages:

- Efficient package management with micromamba
- Small image size compared to full Anaconda
- Compatible with Kaggle's container requirements
- Supports all required scientific packages

### 2.2 Environmental Configuration

The container will support two primary execution modes:

1. **Interactive Mode**: For Jupyter notebook execution on Kaggle
2. **Batch Mode**: For command-line processing of multiple targets

The environment will be configured to auto-detect the execution context and adapt accordingly.

### 2.3 Resource Management

The container will include smart resource management:

- Auto-detection of available CPU cores
- Memory usage monitoring and limitation
- Disk space tracking for large outputs
- Adaptive parallelization based on available resources

### 2.4 Integration Points

The container will provide clear integration points for:

- Input data volumes
- Output feature storage
- Configuration files
- Log access

## 3. Implementation Plan

### 3.1 Dockerfile Enhancements

Starting with the existing Dockerfile, we will make these key enhancements:

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
ENV PATH=/opt/conda/envs/rna3d-core/bin:$PATH

# Add Kaggle-specific optimizations
ENV KAGGLE_CONTAINER=true
ENV PYTHONUNBUFFERED=1

# Copy project files
COPY . .

# Make scripts executable and create directory structure (temporarily switch to root)
USER root
RUN chmod +x scripts/*.sh scripts/*.py && \
    mkdir -p /app/data/processed/thermo_features \
             /app/data/processed/mi_features && \
    # Fix permissions for the data directories
    chown -R $MAMBA_USER:$MAMBA_USER /app/data
# Switch back to the proper non-root user (using the variable from the base image)
USER $MAMBA_USER

# Create volume mount points for data
VOLUME ["/app/data/raw", "/app/data/processed", "/app/logs"]

# Set default parameters
ENV JOBS=1
ENV PF_SCALE=1.5
ENV MEMORY_LIMIT=4.0

# Add script to detect execution environment and set optimal parameters
COPY detect_environment.py /app/
RUN chmod +x /app/detect_environment.py

# Provide dual entrypoints for batch and interactive modes
ENTRYPOINT ["/bin/bash", "-c", "python /app/detect_environment.py && micromamba run -n rna3d-core jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
```

### 3.2 Environment Detection Script

We'll create a `detect_environment.py` script to optimize the container based on the execution environment:

```python
#!/usr/bin/env python3
"""
Environment detection and optimization script for RNA 3D Feature Extractor.
This script auto-detects whether it's running in Kaggle, sets optimal parameters,
and configures the environment accordingly.
"""

import os
import sys
import json
import platform
import psutil

# Detect environment
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

# Get system resources
cpu_count = psutil.cpu_count(logical=False) or 1
memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

# Set optimal parameters
if IS_KAGGLE:
    print("Kaggle environment detected. Optimizing configuration...")
    # Kaggle typically has 4 cores and 16GB RAM for most competitions
    JOBS = min(cpu_count, 4)
    MEMORY_LIMIT = min(memory_gb * 0.7, 12.0)  # 70% of available or max 12GB
else:
    print(f"Local environment detected ({platform.node()}). Optimizing configuration...")
    # Use 80% of available resources for local execution
    JOBS = max(1, cpu_count - 1)
    MEMORY_LIMIT = memory_gb * 0.8

# Write configuration to file
config = {
    "environment": "kaggle" if IS_KAGGLE else "local",
    "resources": {
        "cpu_count": cpu_count,
        "memory_gb": memory_gb,
        "jobs": JOBS,
        "memory_limit": MEMORY_LIMIT
    },
    "platform": {
        "system": platform.system(),
        "node": platform.node(),
        "python": platform.python_version()
    }
}

# Save configuration
with open('/app/environment_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Environment configuration saved: {json.dumps(config, indent=2)}")

# Update environment variables
os.environ['JOBS'] = str(JOBS)
os.environ['MEMORY_LIMIT'] = str(MEMORY_LIMIT)

# Create environment indicator file
if IS_KAGGLE:
    with open('/app/.kaggle_environment', 'w') as f:
        f.write('true')

print("Environment detection and configuration complete.")
```

### 3.3 Dual Execution Modes

To support both interactive and batch processing, we'll create a script for switching between modes:

```bash
#!/bin/bash
# run.sh - Execution mode switcher for RNA 3D Feature Extractor container

MODE=${1:-interactive}
shift

case $MODE in
  interactive)
    echo "Starting in interactive notebook mode..."
    micromamba run -n rna3d-core jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''
    ;;
  batch)
    echo "Starting in batch processing mode..."
    # Get batch parameters
    TARGETS=${1:-"all"}
    FEATURES=${2:-"all"}
    OUTPUT_DIR=${3:-"/app/data/processed"}
    
    echo "Processing targets: $TARGETS"
    echo "Feature types: $FEATURES"
    echo "Output directory: $OUTPUT_DIR"
    
    # Run batch processing
    micromamba run -n rna3d-core python /app/batch_process.py --targets "$TARGETS" --features "$FEATURES" --output-dir "$OUTPUT_DIR"
    ;;
  *)
    echo "Error: Unknown mode '$MODE'"
    echo "Usage: run.sh [interactive|batch] [additional parameters]"
    exit 1
    ;;
esac
```

### 3.4 Kaggle Integration

For seamless Kaggle integration, we'll add:

1. **Kaggle Metadata**: Add Kaggle metadata file for easy importing
2. **Resource Detection**: Adapt to Kaggle's resource constraints automatically
3. **Output Management**: Efficiently handle Kaggle's output storage limitations

Creating a `kaggle-metadata.json` file:

```json
{
  "id": "rna3d-feature-extractor",
  "title": "RNA 3D Feature Extractor",
  "code_file": "notebooks/test_features_extraction.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

### 3.5 Environment Compatibility

To ensure compatibility with both containerized and local (Mamba/Conda) execution:

1. **Environment Detection**: Auto-detect execution environment and adapt accordingly
2. **Conditional Imports**: Use try/except for optional dependencies
3. **Resource Management**: Adapt resource usage based on available capacity
4. **Path Handling**: Use relative paths with proper resolution

## 4. Container Building Process

### 4.1 Build Script

We'll provide a simple build script for creating the container:

```bash
#!/bin/bash
# build.sh - Build RNA 3D Feature Extractor container

VERSION=$(grep -oP 'version = "\K[^"]+' setup.py || echo "dev")
IMAGE_NAME="rna3d-feature-extractor"
TAG="v$VERSION"

echo "Building $IMAGE_NAME:$TAG..."

docker build -t "$IMAGE_NAME:$TAG" -t "$IMAGE_NAME:latest" .

echo "Container build complete: $IMAGE_NAME:$TAG"
echo "To push to a registry: docker push $IMAGE_NAME:$TAG"
```

### 4.2 Volume Management

For efficient data management, we'll use these volume mappings:

```bash
# Example volume mappings
-v $(pwd)/data/raw:/app/data/raw
-v $(pwd)/data/processed:/app/data/processed
-v $(pwd)/logs:/app/logs
```

### 4.3 Configuration

Container configuration will be handled through:

1. **Environment Variables**: For basic configuration
2. **Volume-mounted Config**: For detailed settings
3. **Auto-detected Settings**: For optimal resource allocation

## 5. Runtime Performance Optimizations

### 5.1 Memory Management

To optimize memory usage in containerized environments:

1. **Garbage Collection**: Force garbage collection after processing each batch
2. **Memory Monitoring**: Implement memory limits and monitoring
3. **Chunking**: Process large RNAs in chunks to reduce peak memory

### 5.2 CPU Utilization

For efficient CPU usage:

1. **Auto-scaling**: Detect available cores and scale accordingly
2. **Workload Distribution**: Distribute work evenly across cores
3. **Priority Processing**: Prioritize memory-intensive operations

### 5.3 I/O Optimization

For efficient disk usage:

1. **Buffered I/O**: Use buffered I/O for large files
2. **Compressed Storage**: Keep intermediate results compressed
3. **Selective Persistence**: Only write final outputs to persistent storage

## 6. Container Testing Strategy

### 6.1 Local Testing

Test the container locally with:

```bash
# Run interactive mode
docker run -p 8888:8888 -v $(pwd)/data:/app/data rna3d-feature-extractor:latest

# Run batch mode
docker run -v $(pwd)/data:/app/data rna3d-feature-extractor:latest batch all all
```

### 6.2 Kaggle Testing

To test Kaggle compatibility:

1. Push container to a registry
2. Create a Kaggle dataset with the container reference
3. Test the notebook in Kaggle environment
4. Verify resource usage and outputs

### 6.3 Validation Tests

Validate container functionality with:

1. **Feature Compatibility**: Ensure features match non-containerized execution
2. **Resource Usage**: Verify memory and CPU consumption
3. **Error Handling**: Test error conditions and recovery

## 7. Documentation

### 7.1 Container Usage Guide

```
# RNA 3D Feature Extractor Container Usage

## Interactive Mode
docker run -p 8888:8888 -v $(pwd)/data:/app/data rna3d-feature-extractor

## Batch Mode
docker run -v $(pwd)/data:/app/data rna3d-feature-extractor batch [targets] [features] [output_dir]

## Examples
# Process all targets with all features
docker run -v $(pwd)/data:/app/data rna3d-feature-extractor batch all all

# Process specific targets with specific features
docker run -v $(pwd)/data:/app/data rna3d-feature-extractor batch "R1107,R1108" "thermo,mi"

# Process targets from a file
docker run -v $(pwd)/data:/app/data -v $(pwd)/targets.txt:/app/targets.txt \
  rna3d-feature-extractor batch "/app/targets.txt" all
```

### 7.2 Kaggle Integration Guide

Instructions for using the container in Kaggle:

1. Import the container as a dataset
2. Reference the container in the notebook
3. Configure resources as needed
4. Execute the feature extraction code

## 8. Maintenance Strategy

To ensure ongoing container maintainability:

1. **Version Tagging**: Tag container images with semantic versions
2. **Dependency Updates**: Regular updates of base image and dependencies
3. **Automated Builds**: Implement CI/CD for container builds
4. **Compatibility Testing**: Test across different environments regularly

## 9. Conclusion

This containerization strategy provides a flexible, maintainable approach for deploying the RNA 3D Feature Extractor in both Kaggle and local environments. By focusing on environment detection, resource optimization, and dual execution modes, the container will support a wide range of use cases while maintaining consistent results across environments.