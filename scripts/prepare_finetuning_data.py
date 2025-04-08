import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_alpaca_data, convert_dataset

def main():
    parser = argparse.ArgumentParser(description="Convert Alpaca dataset to LLaMA 2 and Phi formats.")
    parser.add_argument("--alpaca_file", type=str, required=True, help="Path to the Alpaca dataset JSON file.")
    parser.add_argument("--output_llama2", type=str, required=True, help="Output file path for LLaMA 2 format.")
    parser.add_argument("--output_phi", type=str, required=True, help="Output file path for Phi format.")

    args = parser.parse_args()

    # Load the Alpaca dataset
    alpaca_data = load_alpaca_data(args.alpaca_file)

    # Convert and save the dataset
    convert_dataset(alpaca_data, args.output_llama2, args.output_phi)

if __name__ == "__main__":
    main()