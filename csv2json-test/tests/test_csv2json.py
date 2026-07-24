import pytest
import tempfile
import os
from csv2json.main import csv_to_json

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
        expected = '[{"name": "John", "age": "25", "city": "New York"}, {"name": "Jane", "age": "30", "city": "Los Angeles"}, {"name": "Bob", "age": "35", "city": "Chicago"}]'
        assert result == expected
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
        # Should contain indented JSON
        assert '  ' in result
        assert '\n' in result
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
        expected = '{"1": {"name": "John", "age": "25"}, "2": {"name": "Jane", "age": "30"}, "3": {"name": "Bob", "age": "35"}}'
        assert result == expected
    finally:
        os.unlink(temp_file)

def test_malformed_csv():
    """Test handling of malformed CSV"""
    csv_content = """name,age
John,25,extra_column
Jane,30"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        f.flush()
        temp_file = f.name
    
    try:
        # This should not raise an exception, but might produce unexpected results
        # The DictReader should handle this gracefully by treating extra columns
        result = csv_to_json(temp_file)
        # Should still produce valid JSON
        assert result is not None
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
        expected = '[]'
        assert result == expected
    finally:
        os.unlink(temp_file)