#!/bin/bash
# Script to run pseudodihedral angle feature extraction for RNA structures
#
# Usage: ./scripts/run_dihedral_extraction.sh [OPTIONS]
#
# Options:
#   --all                Process all targets in train_labels.csv
#   --batch FILE         Process targets listed in the specified file
#   --target ID          Process a single target
#   --update-metadata    Update feature names and metadata in existing files
#   --limit N            Limit processing to N targets (for testing)
#   --output DIR         Output directory (default: data/processed/dihedral_features)
#   --data-dir DIR       Data directory (default: data/raw)
#
# Examples:
#   # Process all targets
#   ./scripts/run_dihedral_extraction.sh --all
#
#   # Process a single target
#   ./scripts/run_dihedral_extraction.sh --target 1A1T_B
#
#   # Process first 10 targets (for testing)
#   ./scripts/run_dihedral_extraction.sh --all --limit 10
#
#   # Process specific targets from a file
#   ./scripts/run_dihedral_extraction.sh --batch data/target_lists/testing.txt
#
#   # Update metadata in existing files
#   ./scripts/run_dihedral_extraction.sh --update-metadata

# Set default values
OUTPUT_DIR="data/processed/dihedral_features"
DATA_DIR="data/raw"
LIMIT=0
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/dihedral_extraction"
LOG_FILE="${LOG_DIR}/dihedral_extraction_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Parse command line arguments
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --all)
        MODE="all"
        shift
        ;;
        --batch)
        MODE="batch"
        BATCH_FILE="$2"
        shift
        shift
        ;;
        --target)
        MODE="target"
        TARGET_ID="$2"
        shift
        shift
        ;;
        --update-metadata)
        MODE="update-metadata"
        shift
        ;;
        --limit)
        LIMIT="$2"
        shift
        shift
        ;;
        --output)
        OUTPUT_DIR="$2"
        shift
        shift
        ;;
        --data-dir)
        DATA_DIR="$2"
        shift
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done

echo "========== RNA Pseudodihedral Angle Feature Extraction ==========" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Mode: $MODE" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Data directory: $DATA_DIR" | tee -a "$LOG_FILE"
if [ ! -z "$LIMIT" ] && [ "$LIMIT" -gt 0 ]; then
    echo "Limit: $LIMIT" | tee -a "$LOG_FILE"
fi
echo "=============================================" | tee -a "$LOG_FILE"

# Construct the command
CMD="./scripts/extract_pseudodihedral_features.py --output $OUTPUT_DIR --data-dir $DATA_DIR"

# Add mode-specific options
case $MODE in
    all)
    CMD="$CMD --all"
    ;;
    batch)
    CMD="$CMD --batch $BATCH_FILE"
    ;;
    target)
    CMD="$CMD --target $TARGET_ID"
    ;;
    update-metadata)
    CMD="$CMD --update-metadata"
    ;;
    *)
    echo "ERROR: No mode specified. Use --all, --batch, --target, or --update-metadata" | tee -a "$LOG_FILE"
    exit 1
    ;;
esac

# Add limit if specified
if [ ! -z "$LIMIT" ] && [ "$LIMIT" -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Execute the command
echo "Running command: $CMD" | tee -a "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"

# Check if command was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "=============================================" | tee -a "$LOG_FILE"
    echo "Pseudodihedral angle feature extraction completed successfully!" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 0
else
    echo "=============================================" | tee -a "$LOG_FILE"
    echo "ERROR: Pseudodihedral angle feature extraction failed" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi