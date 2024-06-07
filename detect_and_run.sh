#!/bin/bash

# Function to run detection
run_detection() {
    # Arguments for rpicam-detect
    OBJECT_TYPE=$1
    OUTPUT_PREFIX=$2
    OUTPUT_PATTERN="${OUTPUT_PREFIX}%04d.jpg"

    # Run rpicam-detect with specified object type
    rpicam-detect -t 0 -o $OUTPUT_PATTERN --lores-width 400 --lores-height 300 --post-process-file object_detect_tf.json --object $OBJECT_TYPE

    # Check if the object is detected by examining the output file
    if grep -q "OBJECT_DETECTED" detect_results.txt; then
        # Run the Python script if an object is detected
        python3 your_script.py --object $OBJECT_TYPE --output $OUTPUT_PATTERN

        # Record detection
        echo "Recording detection for $OBJECT_TYPE" >> detection_log.txt
    else
        echo "No $OBJECT_TYPE detected."
    fi
}

# Continuous detection loop
while true; do
    run_detection "cat" "cat"
    run_detection "car" "car"
    sleep 2  # Adjust the sleep duration as needed
done
