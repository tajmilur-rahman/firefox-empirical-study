import csv

input_file = "/Users/Mrudhula_1/Desktop/bugs_resolved/bugs_resolved_2.csv"
output_file = "/Users/Mrudhula_1/Desktop/bugs_resolved/bugs_resolved_2_fixed.csv"

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL, doublequote=True)

    for row in reader:
        # Replace internal newlines and carriage returns with a space
        cleaned_row = [field.replace('\n', ' ').replace('\r', ' ') if field else field for field in row]
        writer.writerow(cleaned_row)

print("CSV cleaned, internal quotes doubled, ready for PostgreSQL import!")
