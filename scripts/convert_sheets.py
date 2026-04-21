import pandas as pd

# Path to your XLSX file
file_path = "/Users/Mrudhula/Library/CloudStorage/OneDrive-GannonUniversity/TF_IDF_score_report.xlsx"

# Sheets to convert
sheets_to_convert = ["S1", "S2", "S3", "S4"]

for sheet in sheets_to_convert:
    df = pd.read_excel(file_path, sheet_name=sheet)
    df.to_csv(f"{sheet}.csv", index=False)
    print(f"Converted {sheet} to {sheet}.csv")