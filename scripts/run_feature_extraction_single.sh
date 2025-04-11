#\!/bin/bash
# Test script to run feature extraction on a limited number of RNA sequences
#
# Usage: ./scripts/run_feature_extraction_single.sh [JOBS] [PF_SCALE]
#
# Parameters:
#   JOBS     - Number of parallel jobs (default: number of CPU cores - 1)
#   PF_SCALE - Partition function scaling factor (default: 1.5)
#              Use higher values (1.5-3.0) for longer RNA sequences (1000-3000 nt)
#              to prevent numeric overflow in ViennaRNA calculations

# Check if JOBS argument is provided, otherwise use number of cores - 1
if [ -z "$1" ]; then
    CORES=$(grep -c ^processor /proc/cpuinfo)
    JOBS=$((CORES - 1))
    if [ $JOBS -lt 1 ]; then
        JOBS=1
    fi
else
    JOBS=$1
fi

# Check if PF_SCALE argument is provided, otherwise use default 1.5
PF_SCALE=${2:-1.5}

echo "Starting TEST RNA feature extraction with $JOBS parallel jobs and pf_scale=$PF_SCALE..."
echo "Processing only the first 100 rows of the dataset"

# Make sure the output directory exists
mkdir -p data/processed/features_test

# Run the batch feature runner with parallel processing
python src/data/batch_feature_runner.py \
  --csv data/raw/train_sequences.csv \
  --output-dir data/processed/features_test \
  --id-col target_id \
  --seq-col sequence \
  --max-row 100 \
  --length-min 1 \
  --length-max 1000 \
  --batch-size 20 \
  --jobs $JOBS \
  --output-format both \
  --dynamic-batch \
  --retry 2 \
  --pf-scale $PF_SCALE \
  --verbose

# Convert the results to CSV for easier inspection
echo "Converting NPZ to CSV for inspection..."
python src/data/npz_to_csv.py \
  --input data/processed/features_test/individual/ \
  --output data/processed/features_test/summary.csv \
  --exclude-arrays \
  --verbose

echo "Feature extraction test complete!"
echo "Results saved to data/processed/features_test/"
echo "Summary CSV saved to data/processed/features_test/summary.csv"
echo "Used pf_scale=$PF_SCALE (higher values like 1.5-3.0 help with longer sequences)"
echo "Usage: ./scripts/run_feature_extraction_single.sh [JOBS] [PF_SCALE]"