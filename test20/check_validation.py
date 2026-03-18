import json, sys
from collections import Counter
sys.path.insert(0, '.')
from entities_to_csv import normalize_person_record
from contracts import validate_person

persons = json.loads(open('entities/persons.json', encoding='utf-8').read())
persons = [normalize_person_record(p) for p in persons]
errs_all = []
for p in persons:
    errs = validate_person(p)
    if errs:
        errs_all.extend(errs)

print(f"Total errors: {len(errs_all)}")
# Parse field from error message "'field' value '...' not in vocabulary"
field_counts = Counter()
value_counts = Counter()
for e in errs_all:
    parts = e.split("'")
    if len(parts) >= 4:
        field_counts[parts[1]] += 1
        value_counts[(parts[1], parts[3])] += 1

print("\nError field counts:")
for k, v in field_counts.most_common():
    print(f"  {k}: {v}")

print("\nTop bad values:")
for (field, val), cnt in value_counts.most_common(30):
    print(f"  {field}='{val}': {cnt}")
