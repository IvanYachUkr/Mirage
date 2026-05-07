# Artifact and Workload Summary

This folder contains paper-facing summary tables for the final Test42 100K
benchmark instance and its JOB-family query instantiations.

The query-structure audit compares original templates to final adapted SQL after
normalizing only quoted string literals and numeric constants. Tables, aliases,
joins, operators, predicate positions, aggregate shape, and SQL skeleton must
therefore match exactly.

`query_result_size_summary.csv/.md` reports the distribution of actual
cardinalities for the final nonzero adapted queries. It is intended to show that
the JOB-family instantiations are not merely valid, but span nontrivial result
sizes.
