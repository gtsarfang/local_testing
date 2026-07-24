#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'csv2json'))

from csv2json.main import csv_to_json
import tempfile
import json

def test_basic_conversion():
    """Test basic CSV to JSON conversion"""
    csv_content = """name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        f.flush()
        temp_file = f.name
    
    try:
        result = csv_to_json(temp_file)
        print("Basic conversion test:")
        print(result)
        print()
        return True
    except Exception as e:
        print(f"Error in basic conversion test: {e}")
        return False
    finally:
        os.unlink(temp_file)

def test_pretty_printing():
    """Test pretty-printed JSON output"""
    csv_content = """name,age
John,25
Jane,30"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        f.flush()
        temp_file = f.name
    
    try:
        result = csv_to_json(temp_file, pretty=True)
        print("Pretty printing test:")
        print(result)
        print()
        return True
    except Exception as e:
        print(f"Error in pretty printing test: {e}")
        return False
    finally:
        os.unlink(temp_file)

def test_keyed_output():
    """Test JSON output keyed by a specific column"""
    csv_content = """id,name,age
1,John,25
2,Jane,30
3,Bob,35"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        f.flush()
        temp_file = f.name
    
    try:
        result = csv_to_json(temp_file, key_column='id')
        print("Keyed output test:")
        print(result)
        print()
        return True
    except Exception as e:
        print(f"Error in keyed output test: {e}")
        return False
    finally:
        os.unlink(temp_file)

def test_empty_csv():
    """Test handling of empty CSV file"""
    csv_content = """name,age"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        f.flush()
        temp_file = f.name
    
    try:
        result = csv_to_json(temp_file)
        print("Empty CSV test:")
        print(result)
        print()
        return True
    except Exception as e:
        print(f"Error in empty CSV test: {e}")
        return False
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    print("Testing csv2json functionality:")
    print("=" * 40)
    
    tests = [
        test_basic_conversion,
        test_pretty_printing,
        test_keyed_output,
        test_empty_csv
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"Results: {passed}/{total} tests passed")