import csv
import json
import sys
from typing import List, Dict, Any

def csv_to_json(csv_file_path: str, pretty: bool = False, key_column: str = None) -> str:
    """
    Convert CSV file to JSON.
    
    Args:
        csv_file_path: Path to the CSV file
        pretty: If True, output will be indented
        key_column: If provided, use this column's values as keys for the output object
        
    Returns:
        JSON string
    """
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
        if not rows:
            return json.dumps([] if not key_column else {}, indent=2 if pretty else None)
            
        # If key_column is specified, create a dictionary with that column as keys
        if key_column is not None:
            if key_column not in reader.fieldnames:
                raise ValueError(f"Key column '{key_column}' not found in CSV header")
            
            result = {}
            for row in rows:
                key_value = row[key_column]
                # Remove the key column from the row data
                row_data = {k: v for k, v in row.items() if k != key_column}
                result[key_value] = row_data
        else:
            # Return as array
            result = rows
            
        return json.dumps(result, indent=2 if pretty else None)
        
    except FileNotFoundError:
        raise ValueError(f"File not found: {csv_file_path}")
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")