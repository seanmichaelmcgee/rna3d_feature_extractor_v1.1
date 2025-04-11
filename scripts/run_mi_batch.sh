#!/bin/bash
# Script to run MI feature extraction on a batch of RNA sequences
#
# Usage: ./scripts/run_mi_batch.sh BATCH_FILE [CONFIG]
#
# Parameters:
#   BATCH_FILE - Path to the batch file containing target IDs
#   CONFIG     - (Optional) Configuration profile (default: medium)
#
# Available configurations:
#   small   - For short RNAs with few sequences
#   medium  - Balanced settings (default)
#   large   - For long RNAs or large MSAs
#   memory  - Memory-optimized for limited resources

# Set defaults
MSA_DIR="data/raw/MSA"
OUTPUT_DIR="data/processed/mi_features"
LOG_DIR="logs/mi_extraction"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check for batch file argument
if [ -z "$1" ]; then
    echo "ERROR: Batch file must be specified"
    echo "Usage: $0 BATCH_FILE [CONFIG]"
    exit 1
fi

BATCH_FILE="$1"
BATCH_NAME=$(basename "$BATCH_FILE" .txt)

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Log file for this batch
LOG_FILE="${LOG_DIR}/${BATCH_NAME}_${TIMESTAMP}.log"

# Configuration parameters based on profile
if [ -z "$2" ]; then
    CONFIG="medium"
else
    CONFIG="$2"
fi

case "$CONFIG" in
    small)
        # Small configuration
        MAX_LENGTH=300
        CHUNK_SIZE=250
        OVERLAP=100
        MAX_SEQUENCES=1000
        WORKERS=1
        echo "Using 'small' configuration"
        ;;
    medium)
        # Medium configuration (default)
        MAX_LENGTH=500
        CHUNK_SIZE=400
        OVERLAP=150
        MAX_SEQUENCES=3000
        WORKERS=2
        echo "Using 'medium' configuration"
        ;;
    large)
        # Large configuration
        MAX_LENGTH=750
        CHUNK_SIZE=600
        OVERLAP=200
        MAX_SEQUENCES=5000
        WORKERS=4
        echo "Using 'large' configuration"
        ;;
    memory)
        # Memory-optimized configuration
        MAX_LENGTH=400
        CHUNK_SIZE=300
        OVERLAP=120
        MAX_SEQUENCES=1000
        WORKERS=1
        echo "Using 'memory' configuration (memory-optimized)"
        ;;
    highperf)
        # High-performance configuration for 32-core system
        MAX_LENGTH=750
        CHUNK_SIZE=600
        OVERLAP=200
        MAX_SEQUENCES=5000
        WORKERS=28  # Using 28 of 32 cores to leave some for system operations
        echo "Using 'highperf' configuration (32-core optimization)"
        ;;
    *)
        echo "Unknown configuration: $CONFIG, using 'medium'"
        MAX_LENGTH=500
        CHUNK_SIZE=400
        OVERLAP=150
        MAX_SEQUENCES=3000
        WORKERS=2
        ;;
esac

# Process header
echo "========== RNA MI Feature Extraction ==========" | tee -a "$LOG_FILE"
echo "Batch file: $BATCH_FILE" | tee -a "$LOG_FILE"
echo "Configuration: $CONFIG" | tee -a "$LOG_FILE"
echo "Parameters:" | tee -a "$LOG_FILE"
echo "  MAX_LENGTH: $MAX_LENGTH" | tee -a "$LOG_FILE"
echo "  CHUNK_SIZE: $CHUNK_SIZE" | tee -a "$LOG_FILE"
echo "  OVERLAP: $OVERLAP" | tee -a "$LOG_FILE"
echo "  MAX_SEQUENCES: $MAX_SEQUENCES" | tee -a "$LOG_FILE"
echo "  WORKERS: $WORKERS" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# Count sequences in batch
SEQ_COUNT=$(wc -l < "$BATCH_FILE")
echo "Processing $SEQ_COUNT sequences..." | tee -a "$LOG_FILE"

# Initialize counters
SUCCESSFUL=0
FAILED=0
SKIPPED=0

# Process each sequence in the batch
while IFS= read -r TARGET_ID; do
    # Skip empty lines
    if [ -z "$TARGET_ID" ]; then
        continue
    fi
    
    # Check if the MSA file exists
    MSA_FILE="${MSA_DIR}/${TARGET_ID}.MSA.fasta"
    if [ ! -f "$MSA_FILE" ]; then
        echo "WARNING: MSA file not found for $TARGET_ID, skipping" | tee -a "$LOG_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Check if the output file already exists
    OUTPUT_FILE="${OUTPUT_DIR}/${TARGET_ID}_features.npz"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file already exists for $TARGET_ID, skipping" | tee -a "$LOG_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    echo "Processing $TARGET_ID..." | tee -a "$LOG_FILE"
    
    # Run the MI pipeline
    python src/analysis/rna_mi_pipeline/rna_mi_pipeline.py \
        --input "$MSA_FILE" \
        --output "$OUTPUT_DIR" \
        --max_length $MAX_LENGTH \
        --chunk_size $CHUNK_SIZE \
        --overlap $OVERLAP \
        --max_sequences $MAX_SEQUENCES \
        --workers $WORKERS \
        --parallel \
        --verbose >> "$LOG_FILE" 2>&1
    
    # Check if the output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Successfully processed $TARGET_ID" | tee -a "$LOG_FILE"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "ERROR: Failed to process $TARGET_ID" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done < "$BATCH_FILE"

# Print summary
echo "=============================================" | tee -a "$LOG_FILE"
echo "Batch processing complete!" | tee -a "$LOG_FILE"
echo "Total sequences: $SEQ_COUNT" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESSFUL" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Skipped: $SKIPPED" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Return success if all sequences were processed successfully
if [ $FAILED -eq 0 ]; then
    echo "All sequences processed successfully!" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Some sequences failed to process, check the log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi