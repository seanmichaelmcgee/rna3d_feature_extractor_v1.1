#!/bin/bash
# Test script for feature extraction scripts

echo "=== Testing Feature Extraction Scripts ==="
TEST_DIR="data/script_test"
mkdir -p "$TEST_DIR"

# Function to check if required files exist
check_files() {
  data_type=$1
  target=$2
  
  echo "Checking files for $data_type target $target..."
  
  # Check thermo features
  if [ -f "$TEST_DIR/thermo_features/${target}_thermo_features.npz" ]; then
    echo "✅ Thermo features found"
  else
    echo "❌ Thermo features missing"
  fi
  
  # Check MI features
  if [ -f "$TEST_DIR/mi_features/${target}_mi_features.npz" ]; then
    echo "✅ MI features found"
  else
    echo "❌ MI features missing"
  fi
  
  # Check dihedral features (not for test data)
  if [ "$data_type" != "test" ]; then
    if [ -f "$TEST_DIR/dihedral_features/${target}_dihedral_features.npz" ]; then
      echo "✅ Dihedral features found"
    else
      echo "❌ Dihedral features missing"
    fi
  fi
  
  # Check progress file
  if [ -f "$TEST_DIR/logs/${data_type}_progress.json" ]; then
    echo "✅ Progress file found"
  else
    echo "❌ Progress file missing"
  fi
}

# Clean previous test data if exists
if [ -d "$TEST_DIR" ]; then
  echo "Cleaning previous test data..."
  rm -rf "$TEST_DIR"
  mkdir -p "$TEST_DIR"
fi

# Test train script
echo -e "\n=== Testing Train Feature Extraction ==="
./scripts/feature_extraction/extract_train_features.sh --target 1SCL_A --verbose --output-dir "$TEST_DIR" --report --limit 1

# Check files
check_files "train" "1SCL_A"

# Test validation script
echo -e "\n=== Testing Validation Feature Extraction ==="
./scripts/feature_extraction/extract_validation_features.sh --target R1107 --verbose --output-dir "$TEST_DIR" --report --limit 1

# Check files
check_files "validation" "R1107"

# Test test script
echo -e "\n=== Testing Test Feature Extraction ==="
./scripts/feature_extraction/extract_test_features.sh --target R1107 --verbose --output-dir "$TEST_DIR" --report --limit 1

# Check files
check_files "test" "R1107"

# Test small batch processing
echo -e "\n=== Testing Small Batch Processing ==="
BATCH_DIR="$TEST_DIR/batch_test"
mkdir -p "$BATCH_DIR"

echo "Processing small batch of train data..."
./scripts/feature_extraction/extract_train_features.sh --limit 2 --output-dir "$BATCH_DIR" --verbose

# Count processed files
THERMO_COUNT=$(ls -1 "$BATCH_DIR/thermo_features" 2>/dev/null | wc -l)
DIHEDRAL_COUNT=$(ls -1 "$BATCH_DIR/dihedral_features" 2>/dev/null | wc -l)
MI_COUNT=$(ls -1 "$BATCH_DIR/mi_features" 2>/dev/null | wc -l)

echo "Batch processing results:"
echo "Thermo features: $THERMO_COUNT (should be 2)"
echo "Dihedral features: $DIHEDRAL_COUNT (should be 2)"
echo "MI features: $MI_COUNT (should be 2)"

# Test resume functionality
echo -e "\n=== Testing Resume Functionality ==="
RESUME_DIR="$TEST_DIR/resume_test"
mkdir -p "$RESUME_DIR"

# First pass with 2 targets, setting a small batch size
echo "Starting batch process with 2 targets..."
./scripts/feature_extraction/extract_train_features.sh --limit 2 --output-dir "$RESUME_DIR" --batch-size 1 &
SCRIPT_PID=$!

# Wait 20 seconds then interrupt
echo "Waiting 20 seconds, then interrupting..."
sleep 20
kill -INT $SCRIPT_PID

# Wait a moment for cleanup
sleep 3

# Resume the process
echo "Resuming interrupted process..."
./scripts/feature_extraction/extract_train_features.sh --limit 2 --output-dir "$RESUME_DIR" --resume

# Check number of successful targets in progress file
echo "Checking results..."
if [ -f "$RESUME_DIR/logs/train_progress.json" ]; then
  PROCESSED_COUNT=$(grep -o '"processed_targets"' "$RESUME_DIR/logs/train_progress.json" | wc -l)
  echo "Progress file found with $PROCESSED_COUNT processed targets entries"
  
  # Check actual files
  THERMO_COUNT=$(ls -1 "$RESUME_DIR/thermo_features" 2>/dev/null | wc -l)
  echo "Thermo feature files: $THERMO_COUNT (should be 2 after resume)"
else
  echo "❌ Resume test failed: Progress file not found"
fi

echo -e "\n=== Testing Skip Existing Functionality ==="
SKIP_DIR="$TEST_DIR/skip_test"
mkdir -p "$SKIP_DIR"

# Run first with one target
echo "Processing first target..."
./scripts/feature_extraction/extract_train_features.sh --target 1SCL_A --output-dir "$SKIP_DIR"

# Run again with two targets and skip-existing flag
echo "Processing two targets with skip-existing..."
./scripts/feature_extraction/extract_train_features.sh --limit 2 --output-dir "$SKIP_DIR" --skip-existing

# Check if the first target was skipped
if [ -f "$SKIP_DIR/logs/train_progress.json" ]; then
  SKIPPED_COUNT=$(grep -o '"skipped_targets"' "$SKIP_DIR/logs/train_progress.json" | wc -l)
  echo "Progress file found with $SKIPPED_COUNT skipped targets entries"
  
  # Count actual skipped targets
  SKIPPED_TARGETS=$(grep -o '"skipped_targets": \[[^]]*\]' "$SKIP_DIR/logs/train_progress.json" | grep -o '1SCL_A' | wc -l)
  echo "Skipped targets containing 1SCL_A: $SKIPPED_TARGETS (should be 1)"
else
  echo "❌ Skip test failed: Progress file not found"
fi

echo -e "\n=== Testing Complete ==="
echo "Check $TEST_DIR for all output files"
echo "Test results summary:"
echo "- Single target tests completed"
echo "- Batch processing test completed"
echo "- Resume functionality test completed"
echo "- Skip existing test completed"