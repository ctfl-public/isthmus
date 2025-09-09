#!/usr/bin/env python3
"""
tangle_braces.py - Simple Markdown to Python file tangler
that works with `{file=filename}` syntax.

Usage:
    python tangle_braces.py tutorial.md
"""

import sys
import re
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python tangle_braces.py <markdown_file>")
    sys.exit(1)

md_file = Path(sys.argv[1])

if not md_file.exists():
    print(f"Error: {md_file} does not exist")
    sys.exit(1)

# Regex to match fenced code blocks with {file=filename}
# Matches ```language {file=filename} ... ```
code_block_pattern = re.compile(
    r"```[a-zA-Z0-9]*\s*\{file=([^\}]+)\}\s*\n(.*?)```", re.DOTALL
)

content = md_file.read_text()
matches = code_block_pattern.findall(content)

if not matches:
    print("No code blocks with '{file=...}' found in Markdown")
    sys.exit(0)

# Dictionary to accumulate code for each file
files = {}

for filename, code in matches:
    code = code.rstrip()  # Remove trailing whitespace
    if filename not in files:
        files[filename] = []
    files[filename].append(code)

# Write each file
for filename, blocks in files.items():
    out_path = Path(filename)
    out_path.write_text("\n".join(blocks) + "\n")
    print(f"Wrote {len(blocks)} block(s) to {filename}")
