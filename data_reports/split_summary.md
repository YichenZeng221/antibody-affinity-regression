# ANDD Antibody v2 Stratified Antigen-Level Split Summary

## Design

- Split unit: `antigen_sequence`; an antigen group is assigned to exactly one split.
- Stratification: low/mid/high strata from antigen-level mean target quantiles.
- Requested ratio: 80/10/10; exact row counts can shift slightly because antigen groups are indivisible.
- Tail constraint: both validation and test must include at least one row at or below global P5 and at or above global P95.
- CDR columns are reused from the existing standard `AbNumber + IMGT` annotated dataset for the same `candidate_id` rows.
- No model was trained and the previous split was not overwritten.

- Global target P5: `2.8900`
- Global target P95: `6.5838`

## New Split Target Summary

| split | rows | antigen groups | min | P5 | mean | std | P95 | max | covers global P5 | covers global P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| train | 934 | 568 | 2.0449 | 2.9383 | 4.9473 | 1.1505 | 6.5617 | 8.5343 | True | True |
| val | 117 | 117 | 2.0031 | 2.4199 | 4.9730 | 1.4829 | 6.8475 | 11.8155 | True | True |
| test | 117 | 117 | 2.0349 | 2.8791 | 4.9447 | 1.3705 | 6.7989 | 10.9208 | True | True |

## Original Vs New Val/Test Coverage

| split version | split | rows | min | max | global P5 tail rows | global P95 tail rows |
|---|---|---:|---:|---:|---:|---:|
| original | val | 117 | 2.0938 | 6.9811 | 9 | 4 |
| original | test | 117 | 2.1533 | 8.4692 | 3 | 8 |
| stratified | val | 117 | 2.0031 | 11.8155 | 10 | 9 |
| stratified | test | 117 | 2.0349 | 10.9208 | 7 | 7 |

## Leakage Check

- train vs val antigen overlap: `0`
- train vs test antigen overlap: `0`
- val vs test antigen overlap: `0`

## Next Experiment

Use exactly the existing all-CDR pooled architecture and MSE loss with the new split. Only after training manually should metrics be compared; this isolates split coverage as the changed factor.
