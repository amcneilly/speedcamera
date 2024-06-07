#!/bin/bash

# Argument for object type to detect
OBJECT_TYPE=$1

# Function to run detection
run_detection() {
    # Run rpicam-detect with specified object type
    rpicam-detect --object $OBJECT_TYPE --output detect_results.txt

    # Check if the object is detected by examining the output file
    if grep -q "OBJECT_DETECTED" detect_results.txt; then
        # Run the Python script if an object is detected
        python3 detection.py --arg1 value1 --arg2 value2

        # Record detection
        echo "Recording detection for $OBJECT_TYPE" >> detection_log.txt
    else
        echo "No $OBJECT_TYPE detected."
    fi
}

# Continuous detection loop
while true; do
    run_detection
    sleep 2  # Adjust the sleep duration as needed
done