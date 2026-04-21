import csv
import json

input_csv = '/Users/Mrudhula/PycharmProjects/PythonProject/data/severity_csv_data.csv'
output_csv = '/Users/Mrudhula/PycharmProjects/PythonProject/data/severity_csv_data_fixed.csv'

with open(input_csv, 'r', newline='', encoding='utf-8') as f_in, \
        open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        try:
            # Load conversation text as JSON if possible
            conversation_json = json.loads(row['conversation'])
            # Dump back to string (properly escaped)
            row['conversation'] = json.dumps(conversation_json)
        except json.JSONDecodeError:
            # If conversation is not valid JSON, wrap it as a string
            row['conversation'] = json.dumps(row['conversation'])

        writer.writerow(row)

print("Finished fixing conversation column!")