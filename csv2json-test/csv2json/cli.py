#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from csv2json.main import csv_to_json

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to JSON')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON with indentation')
    parser.add_argument('--key', '-k', help='Use specified column as keys for output object')
    
    args = parser.parse_args()
    
    try:
        json_output = csv_to_json(args.csv_file, pretty=args.pretty, key_column=args.key)
        print(json_output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()