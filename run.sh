#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Example implementations (choose one and modify):

# Example 1: Python implementation
# python3 calculate_reimbursement.py "$1" "$2" "$3"

# Example 2: Node.js implementation
# node calculate_reimbursement.js "$1" "$2" "$3"

# Example 3: Direct bash calculation (for simple logic)
# echo "scale=2; $1 * 100 + $2 * 0.5 + $3" | bc

if [ "$1" == "train" ]; then
    echo "Running cross-validation on training data..."
    python3 calculate_reimbursement.py --train
    exit 0
fi

# Updated to invoke the Python script for reimbursement calculation
python3 calculate_reimbursement.py "$1" "$2" "$3"
