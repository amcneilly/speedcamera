#!/bin/bash

# Function to run detection
run_detection() {
    # Arguments for rpicam-detect
    OBJECT_TYPE=$1
    OUTPUT_PREFIX=$2
    OUTPUT_PATTERN="${OUTPUT_PREFIX}%04d.jpg"
    OUTPUT_FILE="detect_results.txt"

    # Run rpicam-detect with specified object type
    rpicam-detect -t 0 -o $OUTPUT_PATTERN --lores-width 400 --lores-height 300 --post-process-file object_detect_tf.json --object $OBJECT_TYPE > $OUTPUT_FILE

    # Check if the output file exists
    if [ ! -f $OUTPUT_FILE ]; then
        echo "Output file not found!"
        return
    fi

    # Check if the object is detected by examining the output file
    if grep -q "OBJECT_DETECTED" $OUTPUT_FILE; then
        # Run the Python script if an object is detected
        python3 your_script.py --object $OBJECT_TYPE --output $OUTPUT_PATTERN

        # Record detection
        echo "Recording detection for $OBJECT_TYPE" >> detection_log.txt
    else
        echo "No $OBJECT_TYPE detected."
    fi
}

# Ensure the script is run with two arguments
