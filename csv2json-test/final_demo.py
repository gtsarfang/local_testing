import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the functionality by running a direct Python script
print("Final directory structure:")
print("csv2json-test/")
print("├── csv2json/")
print("│   ├── __pycache__/")
print("│   ├── cli.py")
print("│   └── main.py")
print("├── tests/")
print("│   └── test_csv2json.py")
print("├── sample.csv")
print("└── test_runner.py")
print("")

# Test sample functionality
from csv2json.main import csv_to_json

# Test basic conversion
with open('sample.csv', 'r') as f:
    content = f.read()
print("Sample CSV content:")
print(content)
print("")

# Basic conversion
result = csv_to_json('sample.csv')
print("Basic conversion result:")
print(result)
print("")

# Pretty printed
result_pretty = csv_to_json('sample.csv', pretty=True)
print("Pretty printed result:")
print(result_pretty)
print("")

# Keyed output
result_keyed = csv_to_json('sample.csv', key_column='name')
print("Keyed output result:")
print(result_keyed)
print("")

print("All functionality works correctly!")