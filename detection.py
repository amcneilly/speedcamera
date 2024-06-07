import argparse

def main(object_type, output_pattern):
    # Your main code here
    print(f"Object Type: {object_type}, Output Pattern: {output_pattern}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--object', type=str, required=True, help='Type of object detected')
    parser.add_argument('--output', type=str, required=True, help='Output file pattern')
    args = parser.parse_args()
    main(args.object, args.output)
