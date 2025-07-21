import pandas as pd
import json
from pathlib import Path

# Load Excel data
print("Loading Excel data...")
df = pd.read_excel('evaluation_data.xlsx')
print("\nExcel columns:")
print(df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0].to_dict())

# Load promptfoo results if they exist
results_file = Path('promptfoo_results.json')
if results_file.exists():
    print("\nLoading promptfoo results...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    print("\nResults structure:")
    if 'results' in results:
        print("- Has 'results' key")
        if 'table' in results['results']:
            print("- Has 'table' key")
            print(f"- Table length: {len(results['results']['table'])}")
            if len(results['results']['table']) > 0:
                first_row = results['results']['table'][0]
                print("\nFirst result row keys:")
                print(list(first_row.keys()))
else:
    print("\nNo promptfoo results file found") 