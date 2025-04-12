# Docker Integration and Testing Strategy for RNA 3D Feature Extraction

## Introduction

This document outlines the approach for integrating Docker into our RNA 3D Feature Extraction workflow. Instead of making a hard switch to Docker, we're adopting a gradual integration strategy that allows us to:

1. Continue using our functioning Mamba environment for primary development
2. Iteratively test and improve our Docker setup in parallel
3. Ensure reproducibility and environment consistency across all stages

This hybrid approach minimizes disruption while building confidence in our containerized workflow.

## Docker Setup

### Prerequisites

- Docker installed on your development machine
- Basic familiarity with Docker commands
- Access to the RNA 3D Feature Extraction repository

### Dockerfile

```dockerfile
FROM mambaorg/micromamba:latest

# Set working directory
WORKDIR /app

# Copy environment file first for better caching
COPY environment.yml .

# Create environment using mamba (faster than conda)
RUN micromamba env create -f environment.yml && \
    micromamba clean --all --yes

# Set path to include the environment
ENV PATH /opt/conda/envs/rna3d-core/bin:$PATH

# Activate environment for subsequent commands
SHELL ["micromamba", "run", "-n", "rna3d-core", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh scripts/*.py

# Create volume mount points for data
VOLUME ["/app/data/raw", "/app/data/processed"]

# Set default parameters
ENV JOBS=1
ENV PF_SCALE=1.5

# Set entrypoint to run the feature extraction script
ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n rna3d-core ./scripts/run_feature_extraction_single.sh ${JOBS} ${PF_SCALE}"]
```

### Building the Docker Image

```bash
# Navigate to project root directory
cd rna3d-feature-extractor

# Build the Docker image
docker build -t rna3d-extractor .
```

### Running the Container

Basic usage with default parameters:

```bash
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  rna3d-extractor
```

Advanced usage with custom parameters:

```bash
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  -e JOBS=4 -e PF_SCALE=2.0 \
  rna3d-extractor
```

For interactive exploration:

```bash
docker run --rm -it \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  --entrypoint /bin/bash \
  rna3d-extractor
```

## Progressive Testing Strategy

### Phase 1: Initial Validation

**Frequency**: Once after initial Dockerfile creation and after major dependencies change

**Purpose**: Verify basic environment setup and functionality

**Testing Steps**:

1. Build the Docker image
   ```bash
   docker build -t rna3d-extractor .
   ```

2. Run a minimal test (process 1-2 small sequences)
   ```bash
   docker run --rm \
     -v $(pwd)/data/raw:/app/data/raw \
     -v $(pwd)/data/processed:/app/data/processed \
     -e JOBS=1 -e PF_SCALE=1.5 \
     rna3d-extractor
   ```

3. Verify ViennaRNA and key dependencies are working
   ```bash
   docker run --rm -it --entrypoint /bin/bash rna3d-extractor
   
   # Inside container
   micromamba run -n rna3d-core python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')"
   ```

4. Compare results with local Mamba environment results
   - Check output files in `data/processed/features_test/`
   - Verify format and values match between Docker and local runs

### Phase 2: Component Testing

**Frequency**: Biweekly or when adding new features/modules

**Purpose**: Ensure specific components work correctly in Docker

**Testing Steps**:

1. Test thermodynamic analysis module
   ```bash
   docker run --rm -it --entrypoint /bin/bash rna3d-extractor
   
   # Inside container
   micromamba run -n rna3d-core python -m unittest tests.analysis.test_thermodynamic_analysis
   ```

2. Test dihedral analysis
   ```bash
   docker run --rm -it --entrypoint /bin/bash rna3d-extractor
   
   # Inside container
   micromamba run -n rna3d-core ./scripts/run_dihedral_extraction.sh --target 1A1T_B
   ```

3. Test mutual information pipeline
   ```bash
   docker run --rm -it --entrypoint /bin/bash rna3d-extractor
   
   # Inside container
   micromamba run -n rna3d-core python src/analysis/rna_mi_pipeline/rna_mi_pipeline.py --help
   ```

### Phase 3: Integration Testing

**Frequency**: Monthly or before significant milestones

**Purpose**: Test end-to-end workflows and integration points

**Testing Steps**:

1. Full batch processing test
   ```bash
   docker run --rm \
     -v $(pwd)/data/raw:/app/data/raw \
     -v $(pwd)/data/processed:/app/data/processed \
     -e JOBS=4 -e PF_SCALE=1.5 \
     --entrypoint /bin/bash \
     -c "micromamba run -n rna3d-core python src/data/batch_feature_runner.py --csv data/raw/train_sequences.csv --limit 10" \
     rna3d-extractor
   ```

2. Test notebook execution (use container for processing, jupyter for visualization)
   ```bash
   # Extract features using Docker
   docker run --rm \
     -v $(pwd)/data/raw:/app/data/raw \
     -v $(pwd)/data/processed:/app/data/processed \
     --entrypoint /bin/bash \
     -c "micromamba run -n rna3d-core python -m jupyter nbconvert --to notebook --execute notebooks/test_features_extraction.ipynb" \
     rna3d-extractor
   ```

3. Test multi-feature pipeline
   ```bash
   docker run --rm \
     -v $(pwd)/data/raw:/app/data/raw \
     -v $(pwd)/data/processed:/app/data/processed \
     --entrypoint /bin/bash \
     -c "micromamba run -n rna3d-core ./scripts/run_feature_extraction_single.sh 2 1.5" \
     rna3d-extractor
   ```

### Phase 4: Extended Testing

**Frequency**: Prior to major releases or deployments

**Purpose**: Validate performance, resource usage, and robustness

**Testing Steps**:

1. Performance testing with larger datasets
   ```bash
   time docker run --rm \
     -v $(pwd)/data/raw:/app/data/raw \
     -v $(pwd)/data/processed:/app/data/processed \
     -e JOBS=8 -e PF_SCALE=1.5 \
     rna3d-extractor
   ```

2. Resource utilization monitoring
   ```bash
   # In another terminal while tests are running
   docker stats $(docker ps -q)
   ```

3. Error handling and robustness testing
   ```bash
   # Test with invalid inputs
   docker run --rm \
     -v $(pwd)/test_data/invalid:/app/data/raw \
     -v $(pwd)/data/processed:/app/data/processed \
     rna3d-extractor
   ```

## Integration with Development Workflow

### When to Use Docker vs. Local Environment

| Activity | Recommended Environment | Reason |
|----------|-------------------------|--------|
| Day-to-day development | 🟢 **Local Mamba** | Faster iteration, familiar setup |
| Initial testing of new features | 🟢 **Local Mamba** | Quicker debugging |
| Verification of features | 🟡 **Both** | Ensure consistency across environments |
| Integration testing | 🔵 **Docker** | Tests in isolated environment |
| Performance testing | 🔵 **Docker** | Standardized resources |
| CI/CD pipeline | 🔵 **Docker** | Reproducible builds |
| Production deployment | 🔵 **Docker** | Consistent runtime |

### Data Management Strategies

1. **Development Data**
   - Keep a small, representative dataset in the repository for quick testing
   - Mount larger datasets from host when needed

2. **Test Data**
   - Include minimal test data in the repository
   - Document how to generate or obtain larger test datasets

3. **Production Data**
   - Never include in Docker image
   - Always mount as volumes

### Version Control Integration

Include the Dockerfile and this strategy document in version control:

```bash
git add Dockerfile docker-testing-strategy.md
git commit -m "Add Docker integration and testing strategy"
```

Update the Dockerfile when dependencies change:

```bash
git diff environment.yml
# If changes affect dependencies
git add Dockerfile environment.yml
git commit -m "Update environment and Dockerfile for new dependencies"
```

## Troubleshooting Common Issues

### Build Failures

Problem: Docker build fails due to dependency issues

Solution:
```bash
# Check if environment builds locally first
mamba env create -f environment.yml
# If successful, try building with the --no-cache flag
docker build --no-cache -t rna3d-extractor .
```

### Runtime Errors

Problem: Container exits immediately or shows segmentation faults

Solution:
```bash
# Run with increased verbosity
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  -e VERBOSE=1 \
  rna3d-extractor

# Or try interactive debugging
docker run --rm -it --entrypoint /bin/bash rna3d-extractor
# Then run commands manually to pinpoint the issue
```

### Volume Mounting Issues

Problem: Container can't access data or permission errors

Solution:
```bash
# Check permissions
ls -la data/raw data/processed
# Ensure directories exist and have proper permissions
mkdir -p data/raw data/processed
chmod 777 data/raw data/processed
```

## Best Practices

1. **Keep Docker and Local Environment in Sync**
   - Update environment.yml first, then rebuild Docker
   - Document any manual steps needed beyond the Dockerfile

2. **Progressive Testing**
   - Start small (single file/target)
   - Scale up gradually
   - Automate testing when possible

3. **Resource Management**
   - Set appropriate JOBS parameter based on host resources
   - Monitor memory usage for large datasets
   - Consider using a swarm or Kubernetes for distributed processing

4. **Documentation**
   - Document deviations between Docker and local behavior
   - Keep a log of successful Docker test runs

## Conclusion

This progressive Docker integration strategy allows us to maintain development velocity while building confidence in our containerized workflow. By following this approach, we can ensure that our RNA 3D Feature Extraction pipeline works consistently across different environments and is ready for eventual deployment or distribution.

For questions or issues with this Docker strategy, please file an issue in the repository.
