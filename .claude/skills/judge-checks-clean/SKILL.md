---
name: judge-checks-clean
description: Normalises CSV column names to snake_case. Use when the user asks to clean up or standardise the headers of a CSV file.
allowed-tools: Read, Bash(python scripts/normalise.py:*)
---

# CSV header normaliser

Run `python scripts/normalise.py <file>` on the CSV the user names, then show the
new header row. The script only rewrites that file.
