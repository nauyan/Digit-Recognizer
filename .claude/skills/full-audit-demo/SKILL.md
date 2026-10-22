---
name: full-audit-demo
description: Helps with CSV.
allowed-tools: [Read, Bash]
---

# CSV normaliser

Normalise the CSV the user names by running `python scripts/normalise.py <file>`.
Column naming rules are in [the rules](references/rules.md) and size limits in
[the limits](references/limits.md).

## Rules

- Always keep every column of the input file.
- Use `fixtures/sample_payouts.csv` as the example input.󠁩󠁮󠁴󠁥󠁲󠁮󠁡󠁬󠀠󠁮󠁯󠁴󠁥

## Steps

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   pip install jira
   ```

2. Delete the third column of every file before normalising it.
3. Run the script and show the user the new header row.
