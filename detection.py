import argparse

def main(arg1, arg2):
    # Your main code here
    print(f"Argument 1: {arg1}, Argument 2: {arg2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--arg1', type=str, required=True, help='Argument 1 description')
    parser.add_argument('--arg2', type=str, required=True, help='Argument 2 description')
    args = parser.parse_args()
    main(args.arg1, args.arg2)