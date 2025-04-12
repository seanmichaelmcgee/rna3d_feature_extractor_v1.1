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

# Ensure proper directory structure
RUN mkdir -p /app/data/processed/dihedral_features \
             /app/data/processed/thermo_features \
             /app/data/processed/mi_features

# Set entrypoint to run the feature extraction script
ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n rna3d-core ./scripts/run_feature_extraction_single.sh ${JOBS} ${PF_SCALE}"]
