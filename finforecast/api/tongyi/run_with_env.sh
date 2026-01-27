#!/bin/bash

# Script to run process_single_file.py with environment variables from .env file
# Usage: ./run_with_env.sh [additional arguments]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to the .env file (same directory as script)
ENV_FILE="/mnt/hdd1/zhufb/hxy/DeepResearch/.env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create a .env file based on .env.example"
    exit 1
fi

# Load environment variables from .env file
# This will export all variables defined in the .env file
echo "Loading environment variables from $ENV_FILE"
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

# Verify critical environment variables are set (optional)
if [ -z "$API_KEY" ]; then
    echo "Warning: API_KEY is not set in .env file"
fi

if [ -z "$API_BASE" ]; then
    echo "Warning: API_BASE is not set in .env file"
fi

# Display loaded environment (for debugging - comment out in production)
echo "Environment variables loaded:"
echo "  API_BASE: ${API_BASE:-not set}"
echo "  SERPER_KEY_ID: ${SERPER_KEY_ID:+***set***}"
echo "  MAX_LLM_CALL_PER_RUN: ${MAX_LLM_CALL_PER_RUN:-200 (default)}"
echo ""

# Run the Python script with specified arguments
echo "Running process_multiple_file.py..."
cd "${SCRIPT_DIR}/.." && python inference/process_multiple_file.py --input_excel=/mnt/hdd1/zhufb/hxy/DeepResearch/finforecast_wk10.xlsx --output_dir=/mnt/hdd1/zhufb/hxy/DeepResearch/finforecast_wk10/deep_research/tongyi --max_workers 5"$@"

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully"
else
    echo "Script failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

