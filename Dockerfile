FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean --all --yes

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
ENTRYPOINT ["/bin/bash", "-c", "conda run -n rna3d-core ./scripts/run_feature_extraction_single.sh ${JOBS} ${PF_SCALE}"]
