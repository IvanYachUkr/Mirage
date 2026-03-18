"""
fix_unicode_prints.py
Replaces non-ASCII Unicode special characters in Python source files
with ASCII equivalents so they survive Windows cp1252 terminal encoding.
Only rewrites files that actually change.
"""
import os
import re
import pathlib

# Map of Unicode chars commonly used in print statements -> ASCII replacements
REPLACEMENTS = {
    '\u2192': '->',    # -> right arrow
    '\u2190': '<-',    # <- left arrow
    '\u2248': '~=',    # ~= approximately equal
    '\u2260': '!=',    # != not equal
    '\u2264': '<=',    # <= less than or equal
    '\u2265': '>=',    # >= greater than or equal
    '\u00d7': 'x',     # x multiplication sign
    '\u00f7': '/',     # / division sign
    '\u2014': '--',    # -- em dash
    '\u2013': '-',     # - en dash
    '\u2026': '...',   # ... ellipsis
    '\u2022': '*',     # * bullet
    '\u25b6': '>',     # > play button
    '\u2714': 'OK',    # OK check mark
    '\u2718': 'X',     # X cross mark
    '\u2728': '*',     # * sparkles
    '\u00b7': '.',     # . middle dot
    '\u03b1': 'alpha', # alpha
    '\u03b2': 'beta',  # beta
    '\u03b3': 'gamma', # gamma
    '\u03c3': 'sigma', # sigma
    '\u03bc': 'mu',    # mu
    '\u221e': 'inf',   # inf
    '\u221a': 'sqrt',  # sqrt
    '\u03a3': 'Sigma', # Sigma
    '\u0394': 'Delta', # Delta
    '\u0393': 'Gamma', # Gamma
}

BASE = pathlib.Path(__file__).parent
changed = []
for py_file in BASE.rglob('*.py'):
    if '__pycache__' in str(py_file):
        continue
    try:
        text = py_file.read_text(encoding='utf-8')
    except Exception:
        continue
    new_text = text
    for uni, asc in REPLACEMENTS.items():
        new_text = new_text.replace(uni, asc)
    if new_text != text:
        py_file.write_text(new_text, encoding='utf-8')
        changed.append(str(py_file.relative_to(BASE)))

print(f"Fixed {len(changed)} files:")
for f in changed:
    print(f"  {f}")
