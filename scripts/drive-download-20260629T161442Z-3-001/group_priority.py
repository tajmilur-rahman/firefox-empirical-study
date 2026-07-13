import csv
import re
from collections import defaultdict, Counter

csv.field_size_limit(10**7)

# keyphrase -> priority -> count
keyphrase_priority_count = defaultdict(lambda: Counter())

with open('priority_bugs_labeled_bertscore.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        priority = row['priority']
        labels = [l.strip() for l in row['matched_labels'].split(';')]
        for label in labels:
            if label:
                keyphrase_priority_count[label][priority] += 1

# write to csv
with open('keyphrase_priority_counts.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['keyphrase', 'P1', 'P2', 'P3', 'P4', 'P5'])
    for phrase, counts in sorted(keyphrase_priority_count.items()):
        writer.writerow([
            phrase,
            counts.get('P1', 0),
            counts.get('P2', 0),
            counts.get('P3', 0),
            counts.get('P4', 0),
            counts.get('P5', 0),
        ])

print("Done. Output: keyphrase_priority_counts.csv")