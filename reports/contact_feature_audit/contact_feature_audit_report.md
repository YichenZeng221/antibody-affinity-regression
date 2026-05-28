# ANDD Antibody v2 Stratified Contact / Interface Feature Availability Audit

## Scope

- 目标：检查现有结构和 metadata 是否足以支持下一步 contact/interface feature extraction。
- 数据：`data/processed_affinity/expanded_affinity_antibody_v2_stratified/{train,val,test}.csv`。
- 本次只读取数据和 PDB chain ID；**没有**批量计算原子距离、没有训练模型、没有修改 dataset。
- 外部 SAbDab structure archive 仅通过原绝对路径只读访问，没有复制 31GB 结构目录。

## Inputs Found

- ANDD stratified rows: **1168** (1064 unique `pdb_id`).
- SAbDab summary metadata: `data/raw/sabdab_summary.tsv`，可提供 `Hchain`, `Lchain`, `antigen_chain` 候选。
- Project-local cached PDB files: `data/pdb` = 655 files; **0 / 1168** ANDD rows match this local cache.
- External SAbDab archive root exists: `True` at `/Users/yichenzeng/Downloads/all_structures`.
- External `raw/imgt/chothia` PDB counts: 10780 / 10779 / 10780.
- External `raw/imgt/chothia` matches to ANDD rows: **1168 / 1168 / 1168**.

### Relevant Files and Code Located

- `data/raw/sabdab_summary.tsv`: SAbDab structure metadata and chain candidates.
- `data/processed_affinity/sabdab_structure_archive_inspection/`: previous archive existence/chain audit.
- `scripts/inspect_sabdab_structure_archive.py`: prior lightweight PDB archive inspection.
- `scripts/build_sabdab_chain_dataset.py`: existing PDB/chain sequence parsing utility pattern.
- `scripts/analyze_andd_stratified_model_fit.py`: already reserves optional `contact_count`, `min_distance`, and `interface_residue_count`, but reported them missing.
- `src/affinity_interaction_model.py` and `src/affinity_cross_attention_model.py`: sequence-level interaction models; neither consumes 3D contact geometry.

## Availability Summary

| split | rows | structure file found | unambiguous viable H/L/antigen mapping | basic interface feature-ready | CDR-contact pipeline candidates* | rows joinable to existing predictions |
|---|---:|---:|---:|---:|---:|---:|
| train | 934 | 934 | 360 | 360 | 360 | 934 |
| val | 117 | 117 | 54 | 54 | 54 | 117 |
| test | 117 | 117 | 58 | 58 | 58 | 117 |

\* `CDR-contact pipeline candidates` means structure + one viable chain mapping + IMGT file + standard CDR annotation are present. CDR-to-structure residue mapping still needs validation.

### Overall Counts

- Samples with a PDB/structure file mapping: **1168 / 1168**.
- Samples with at least one complete H/L/antigen chain metadata option: **1167 / 1168**.
- Samples with an unambiguous viable H/L/antigen chain mapping in the raw structure: **472 / 1168**.
- Samples whose chain metadata remain ambiguous because multiple viable mappings exist: **695 / 1168**.
- Samples with no viable complete chain mapping found in the raw structure: **1 / 1168**.
- Samples ready for a conservative first pass of basic interface features: **472 / 1168**.
- Samples potentially usable for CDR-level contact features after residue-mapping validation: **472 / 1168**.
- Samples already joinable to at least one existing prediction/residual file, including prior fit-diagnosis inference: **1168 / 1168**.

## Feature-by-Feature Feasibility

| feature | availability now | required next step |
|---|---|---|
| antibody-antigen contact count | basic extraction prerequisites present for 472 rows | choose a distance threshold and compute atom/residue contacts for unambiguous chain mappings |
| minimum antibody-antigen distance | basic extraction prerequisites present for 472 rows | compute minimum heavy/light-to-antigen atomic distance |
| interface residue count | basic extraction prerequisites present for 472 rows | define interface cutoff and count residues touching across chains |
| CDR-antigen contact count | 472 candidate rows, not validated yet | map IMGT CDR residues to PDB residues before counting |
| HCDR3 contact fraction | 472 candidate rows, not validated yet | validate heavy-chain IMGT numbering/residue alignment |
| LCDR3 contact fraction | 472 candidate rows, not validated yet | validate light-chain IMGT numbering/residue alignment |

## What Is Missing or Not Yet Validated

- `PDB ID`: present for all rows and externally mapped to structure files.
- `Hchain/Lchain/antigen_chain`: not stored in the stratified CSV itself; recoverable as SAbDab summary candidates, but multiple viable chain mappings remain for a substantial subset.
- `Antigen chain mapping`: included in the SAbDab candidate mappings, subject to the same ambiguity check.
- `CDR residue annotation`: sequence-level AbNumber + IMGT CDR fields are already present.
- `Sequence-to-structure residue mapping`: **not yet validated**. This is the central missing step before trustworthy CDR-level contact fractions can be generated.

## Can Contact Features Be Joined to Existing Errors?

- Yes: predictions and dataset rows share `sample_id`; currently **1168** rows already have a matching prediction from at least one saved evaluation or fit-diagnosis output.
- Existing fit-diagnosis predictions cover train/val/test for pooled and cross-attention models; saved tail-aware checkpoint predictions currently provide test-set residuals.
- Once contact features are extracted, test rows can be merged with residuals to analyze `contact feature vs target`, `contact feature vs absolute_error`, and `contact feature vs tail error`.
- No correlation is computed here because contact values have not yet been extracted.

## Minimal Viable Contact Feature Pipeline

1. Start only with rows having one viable H/L/antigen chain mapping and an available raw/IMGT structure.
2. Validate chain sequences and map the AbNumber/IMGT CDR residues onto structure residue numbers; keep ambiguous or mismatched rows flagged rather than forcing an assignment.
3. Compute simple geometry features first: antibody-antigen contact count, minimum distance, interface residue count, CDR-antigen contact count, HCDR3/LCDR3 contact fractions.
4. Merge feature tables to the existing prediction CSVs by `sample_id` and audit whether these features explain tail errors or prediction compression.
5. Only after the audit shows signal should contact/interface features enter a new modeling experiment.

## Honest Conclusion

The structure resource is promising: all 1168 stratified samples have an external SAbDab structure-file match. However, contact modeling is not immediately ready for every row. A conservative basic-interface pilot can begin with 472 rows whose chain mapping is unambiguous in the available metadata and structure. CDR-level contact features require an additional residue-mapping validation step; until that is done, treating CDR contacts as ground truth would be unsafe.

## Outputs

- Availability table: `outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_availability.csv`
- This report: `outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_audit_report.md`
