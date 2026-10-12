import csv
import sys

path = sys.argv[1]
with open(path, newline='') as handle:
    rows = list(csv.reader(handle))
rows[0] = [column.strip().lower().replace(' ', '_') for column in rows[0]]
with open(path, 'w', newline='') as handle:
    csv.writer(handle).writerows(rows)
