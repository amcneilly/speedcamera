#!/bin/bash

# Function to run detection
run_detection() {
    # Arguments for rpicam-detect
    OBJECT_TYPE=$1
    OUTPUT_PREFIX=$2
    OUTPUT_PATTERN="${OUTPUT_PREFIX}%04d.jpg"
    DETECT_RESULTS="detect_results.txt"

    # Run rpicam-detect with specified object type
    rpicam-detect -t 0 -o $OUTPUT_PATTERN --lores-width 400 --lores-height 300 --post-process-file object_detect_tf.json --object $OBJECT_TYPE --output $DETECT_RESULTS

    # Check if the detect_results.txt file exists
    if [ ! -f $DETECT_RESULTS ]; then
        echo "Detection results file not found!"
        return
    fi

    # Check if the object is detected by examining the output file
    if grep -q "OBJECT_DETECTED" $DETECT_RESULTS; then
        # Run the Python script if an object is detected
        python3 your_script.py --object $OBJECT_TYPE --output $OUTPUT_PATTERN

        # Record detection
        echo "Recording detection for $OBJECT_TYPE" >> detection_log.txt
    else
        echo "No $OBJECT_TYPE detected."
    fi

    # Clean up the detection results file
    rm $DETECT_RESULTS
}

# Ensure the script is run with two arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <object_type> <output_prefix>"
    exit 1
fi

# Set the OBJECT_TYPE and OUTPUT_PREFIX variables from command-line arguments
OBJECT_TYPE=$1
OUTPUT_PREFIX=$2

# Continuous detection loop
while true; do
    run_detection $OBJECT_TYPE $OUTPUT_PREFIX
    sleep 2  # Adjust the sleep duration as needed
done
