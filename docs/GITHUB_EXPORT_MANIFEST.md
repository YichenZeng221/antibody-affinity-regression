# GitHub Export Manifest

This `github_ready/` directory is a curated export of the local project. It is designed for GitHub sharing and presentation, while keeping large or local-only artifacts out of version control.

## Included

- Core implementation:
  - `src/`
  - `scripts/`
  - selected `run_train_*.py` entry points
- Selected configs:
  - ANDD antibody v2 stratified pooled baseline
  - cross-attention baseline
  - tail-aware w2 configs
  - multi-seed configs
- Final reports:
  - English final project summary
  - final results index
  - compact ANDD stratified model summary
  - compact contact/interface audit summary
- Final presentation figures:
  - prediction compression
  - residual trend
  - multi-seed tail-aware w2
  - CDR3 contact augmentation
  - contact availability funnel
  - CDR mapping validation
- Lightweight summary CSV / MD reports:
  - fit diagnosis metrics
  - multi-seed summary
  - contact feature audit tables
  - dataset split summaries

## Excluded

- Virtual environments: `.venv/`, `.venv_abnumber/`, `.venv_tdc/`
- Raw datasets: ANDD xlsx, SAbDab summary/raw files
- Structure archives and PDB files
- Model checkpoints
- Large prediction dumps
- Local archive/debug outputs

## Portfolio Cleanup Decisions

- Kept all `src/` and `scripts/` files to show the full research workflow.
- Kept only four final configs instead of all intermediate seed/sweep configs.
- Kept final reports, final figures, and lightweight summary tables.
- Removed Python cache files from this export.
- Excluded raw data, checkpoints, PDB files, and large prediction dumps.

## Suggested GitHub Workflow

From inside `github_ready/`:

```bash
git init
git add .
git commit -m "Add antibody-antigen affinity regression project export"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Before pushing, review the file list:

```bash
find . -type f | sort
du -sh .
```
