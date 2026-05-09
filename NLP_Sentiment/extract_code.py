import json
import os

notebook_path = r"C:\Users\narla\Downloads\123_project_nlp.ipynb"
output_path = r"C:\Users\narla\.gemini\antigravity\scratch\nlp_sentiment_frontend\extracted_code.py"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']

    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in code_cells:
            if isinstance(cell, list):
                f.write("".join(cell))
            else:
                f.write(cell)
            f.write("\n\n")
    print(f"Successfully extracted code to {output_path}")
except Exception as e:
    print(f"Error: {e}")
