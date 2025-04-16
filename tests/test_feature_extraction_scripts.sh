#!/bin/bash
# Comprehensive test runner for feature extraction scripts
# This script runs various tests on the feature extraction scripts
# and generates a report of the results.

# Set up test directory
TEST_DIR="data/test_output"
REPORT_FILE="tests/feature_extraction_test_report.md"
TEST_LOG="tests/feature_extraction_test.log"

# Create directories
mkdir -p "$TEST_DIR"
mkdir -p "$(dirname "$REPORT_FILE")"

# Start test log
echo "Running feature extraction script tests at $(date)" | tee "$TEST_LOG"
echo "==================================================" | tee -a "$TEST_LOG"

# Initialize report
cat > "$REPORT_FILE" << EOL
# Feature Extraction Scripts Test Report

Test run on: $(date)

## Test Environment

- Host: $(hostname)
- User: $(whoami)
- Working directory: $(pwd)
- RNA3D Feature Extractor Version: v1.1

## Tests Run

EOL

# Function to run a test and update the report
run_test() {
    TEST_NAME="$1"
    TEST_CMD="$2"
    TEST_DESC="$3"
    
    echo -e "\n>> Running test: $TEST_NAME" | tee -a "$TEST_LOG"
    echo "   $TEST_DESC" | tee -a "$TEST_LOG"
    echo "   Command: $TEST_CMD" | tee -a "$TEST_LOG"
    
    # Add to report
    echo "### $TEST_NAME" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Description:** $TEST_DESC" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Command:**" >> "$REPORT_FILE"
    echo '```bash' >> "$REPORT_FILE"
    echo "$TEST_CMD" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Run the test and capture output
    START_TIME=$(date +%s)
    TEST_OUTPUT=$(eval "$TEST_CMD" 2>&1)
    RESULT=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Log the output
    echo "$TEST_OUTPUT" >> "$TEST_LOG"
    
    # Add result to report
    if [ $RESULT -eq 0 ]; then
        echo "**Result:** ✅ PASS" >> "$REPORT_FILE"
    else
        echo "**Result:** ❌ FAIL" >> "$REPORT_FILE"
    fi
    
    echo "**Duration:** ${DURATION}s" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Output:**" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "$TEST_OUTPUT" | head -n 20 >> "$REPORT_FILE"
    if [ $(echo "$TEST_OUTPUT" | wc -l) -gt 20 ]; then
        echo "... (output truncated, see full log)" >> "$REPORT_FILE"
    fi
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    return $RESULT
}

# Main test section

# Test 1: Basic train feature extraction
TEST_DIR1="$TEST_DIR/train_basic"
mkdir -p "$TEST_DIR1"
run_test "Basic Train Feature Extraction" \
    "./scripts/feature_extraction/extract_train_features.sh --target 1SCL_A --output-dir $TEST_DIR1 --limit 1" \
    "Extract features for a single training target"

# Test 2: Basic validation feature extraction
TEST_DIR2="$TEST_DIR/validation_basic"
mkdir -p "$TEST_DIR2"
run_test "Basic Validation Feature Extraction" \
    "./scripts/feature_extraction/extract_validation_features.sh --target R1107 --output-dir $TEST_DIR2 --limit 1" \
    "Extract features for a single validation target"

# Test 3: Basic test feature extraction
TEST_DIR3="$TEST_DIR/test_basic"
mkdir -p "$TEST_DIR3"
run_test "Basic Test Feature Extraction" \
    "./scripts/feature_extraction/extract_test_features.sh --target R1107 --output-dir $TEST_DIR3 --limit 1" \
    "Extract features for a single test target"

# Test 4: Verify feature output format
if [ -d "$TEST_DIR1" ] && [ "$(ls -A "$TEST_DIR1/thermo_features" 2>/dev/null)" ]; then
    run_test "Verify Feature Format" \
        "./scripts/feature_extraction/verify_features.py $TEST_DIR1" \
        "Verify the format of extracted features"
fi

# Test 5: Test resume functionality
TEST_DIR5="$TEST_DIR/resume_test"
mkdir -p "$TEST_DIR5"
# First run with interrupt
(./scripts/feature_extraction/extract_train_features.sh --limit 3 --output-dir "$TEST_DIR5" --batch-size 1 > "$TEST_DIR5/first_run.log" 2>&1) &
PID=$!
sleep 15
kill -INT $PID
sleep 3
# Now run with resume
run_test "Resume Functionality" \
    "./scripts/feature_extraction/extract_train_features.sh --limit 3 --output-dir $TEST_DIR5 --resume" \
    "Test ability to resume an interrupted extraction"

# Test 6: Skip existing functionality
TEST_DIR6="$TEST_DIR/skip_test"
mkdir -p "$TEST_DIR6"
# First run with one target
./scripts/feature_extraction/extract_train_features.sh --target 1SCL_A --output-dir "$TEST_DIR6" > "$TEST_DIR6/first_run.log" 2>&1
# Now run with skip-existing
run_test "Skip Existing Files" \
    "./scripts/feature_extraction/extract_train_features.sh --limit 2 --output-dir $TEST_DIR6 --skip-existing" \
    "Test ability to skip targets with existing feature files"

# Test 7: Multiple targets batch processing
TEST_DIR7="$TEST_DIR/batch_test"
mkdir -p "$TEST_DIR7"
run_test "Batch Processing" \
    "./scripts/feature_extraction/extract_train_features.sh --limit 2 --output-dir $TEST_DIR7" \
    "Process multiple targets in batch mode"

# Test 8: Generate report
TEST_DIR8="$TEST_DIR/report_test"
mkdir -p "$TEST_DIR8"
run_test "HTML Report Generation" \
    "./scripts/feature_extraction/extract_train_features.sh --target 1SCL_A --output-dir $TEST_DIR8 --report" \
    "Test HTML report generation"

# Add summary to report
echo "## Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- Total tests run: 8" >> "$REPORT_FILE"
SUCCESS_COUNT=$(grep -c "✅ PASS" "$REPORT_FILE")
FAIL_COUNT=$(grep -c "❌ FAIL" "$REPORT_FILE")
echo "- Tests passed: $SUCCESS_COUNT" >> "$REPORT_FILE"
echo "- Tests failed: $FAIL_COUNT" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ $FAIL_COUNT -eq 0 ]; then
    echo "**Overall Status: PASS** ✅" >> "$REPORT_FILE"
else
    echo "**Overall Status: FAIL** ❌" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## Notes" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- Full test logs are available in $TEST_LOG" >> "$REPORT_FILE"
echo "- Test output files are in $TEST_DIR" >> "$REPORT_FILE"
echo "- These tests verify basic functionality and do not guarantee correctness of extracted features" >> "$REPORT_FILE"

echo -e "\nTest run complete!" | tee -a "$TEST_LOG"
echo "Test report generated at: $REPORT_FILE" | tee -a "$TEST_LOG"
echo "See $TEST_LOG for full details" | tee -a "$TEST_LOG"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "\n✅ All tests passed!" | tee -a "$TEST_LOG"
    exit 0
else
    echo -e "\n❌ $FAIL_COUNT tests failed." | tee -a "$TEST_LOG"
    exit 1
fi