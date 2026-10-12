---
name: full-audit-clean
description: Normalises CSV column names to snake_case. Use when the user asks to clean up or standardise the headers of a CSV file.
allowed-tools: Read, Bash(python scripts/normalise.py:*)
---

# CSV header normaliser

Normalise the headers of the CSV the user names by running
`python scripts/normalise.py <file>`. Naming rules are in [the rules](references/rules.md).

## Steps

1. Run the script on the file. It uses only the Python standard library and
   rewrites only the header row of that file.
2. Show the user the new header row.

`fixtures/sample.csv` is example input with sample values only.
