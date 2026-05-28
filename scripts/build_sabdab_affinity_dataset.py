"""Build a sequence-only antibody-antigen affinity regression dataset.

:
 Stage 1 regression MVP 
 heavy/light classification  data/processed/ 

raw SAbDab summary ?
SAbDab summary TSV ,
 PDB IDheavy chain IDlight chain IDantigen chain ID
affinity 

PDB ?
summary  PDB chain, amino acid sequence
 PDB ID / .pdb , chain id 

:
    data/raw/sabdab_summary.tsv
    data/pdb/*.pdb

:
    data/processed_affinity/sequence_only/train.csv
    data/processed_affinity/sequence_only/val.csv
    data/processed_affinity/sequence_only/test.csv

:
    input  = heavy_sequence + light_sequence + antigen_sequence
    target = neg_log10_affinity = -log10(affinity)

 -log10(affinity)?
affinity , 1e-12  1e-4
 -log10 ,,
 1e-9  9
"""

from pathlib import Path
import argparse
import math
import random
import urllib.request

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


REQUIRED_COLUMNS = [
    "pdb",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "antigen_type",
    "antigen_name",
    "affinity",
    "delta_g",
    "affinity_method",
    "temperature",
    "pmid",
]

OUTPUT_COLUMNS = [
    "sample_id",
    "pdb",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "antigen_type",
    "antigen_name",
    "heavy_sequence",
    "light_sequence",
    "antigen_sequence",
    "affinity",
    "neg_log10_affinity",
    "delta_g",
    "affinity_method",
    "temperature",
    "pmid",
]


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

    --max_rows  optional:
    - : rows
    -  100: 100 ,
    """

    parser = argparse.ArgumentParser(description="Build SAbDab affinity regression CSV files.")
    parser.add_argument("--summary_path", default="data/raw/sabdab_summary.tsv")
    parser.add_argument("--pdb_dir", default="data/pdb")
    parser.add_argument("--output_dir", default="data/processed_affinity/sequence_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--min_sequence_length", type=int, default=20)
    return parser.parse_args()


def is_missing(value: str) -> bool:
    """Return True if a SAbDab field is missing."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def clean_chain_id(chain_id: str) -> str:
    """Remove surrounding whitespace from a chain id."""

    return str(chain_id).strip()


def split_chain_ids(chain_text: str) -> list[str]:
    """Split antigen_chain values like 'A | B' or 'A|B'.

    :
     antigen  chain, chain 
    SAbDab :
        A | B
     ["A", "B"],,
    """

    return [part.strip() for part in str(chain_text).split("|") if part.strip()]


def download_pdb_if_needed(pdb_id: str, pdb_dir: Path, stats: dict) -> Path | None:
    """Use cached PDB if available, otherwise download it from RCSB.

    :
     PDB , data/pdb/ ,
    
    """

    pdb_id = pdb_id.upper()
    pdb_path = pdb_dir / f"{pdb_id}.pdb"

    if pdb_path.exists():
        stats["pdb_cache_reused"] += 1
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  downloading PDB: {pdb_id}")

    try:
        urllib.request.urlretrieve(url, pdb_path)
        stats["pdb_downloaded"] += 1
        return pdb_path
    except Exception as error:
        print(f"  WARNING: failed to download {pdb_id}: {error}")
        stats["failed_downloads"] += 1
        return None


def load_structure(pdb_path: Path, pdb_id: str, stats: dict):
    """Parse one PDB file with Biopython.

    Biopython  PDBParser  .pdb  Python :
    structure -> model -> chain -> residue
     chain id 
    """

    parser = PDBParser(QUIET=True)

    try:
        return parser.get_structure(pdb_id, pdb_path)
    except Exception as error:
        print(f"  WARNING: failed to parse {pdb_path}: {error}")
        stats["failed_pdb_parse"] += 1
        return None


def extract_chain_sequence(structure, pdb_id: str, chain_id: str, min_length: int, stats: dict) -> str | None:
    """Extract one chain sequence from a parsed PDB structure.

    :
    Hchain / Lchain / antigen_chain  chain id, HLA
     amino acid sequence  PDB  residue 

     chain,
    """

    model = structure[0]

    if chain_id not in model:
        print(f"  WARNING: {pdb_id} does not contain chain {chain_id}")
        stats["failed_sequence_extractions"] += 1
        return None

    amino_acids = []

    for residue in model[chain_id]:
        if not is_aa(residue, standard=True):
            continue
        amino_acids.append(seq1(residue.get_resname()))

    sequence = "".join(amino_acids)

    if len(sequence) < min_length:
        stats["skipped_short_sequences"] += 1
        return None

    return sequence


def is_valid_antigen_type(antigen_type: str) -> bool:
    """Keep protein/peptide antigens and skip haptens.

    Hapten ,/
     Stage 1  sequence, protein / peptide antigen
    """

    text = str(antigen_type).lower()
    has_sequence_type = "protein" in text or "peptide" in text
    is_hapten = "hapten" in text
    return has_sequence_type and not is_hapten


def split_by_pdb(records: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records by PDB ID so one PDB cannot appear in multiple splits.

    :
     PDB  train  test,
    ,
    PDB-level split  row random split 
    """

    random.seed(seed)

    pdb_to_records = {}
    for record in records:
        pdb_to_records.setdefault(record["pdb"], []).append(record)

    pdb_ids = list(pdb_to_records.keys())
    random.shuffle(pdb_ids)

    train_end = int(len(pdb_ids) * 0.8)
    val_end = int(len(pdb_ids) * 0.9)

    train_pdbs = set(pdb_ids[:train_end])
    val_pdbs = set(pdb_ids[train_end:val_end])
    test_pdbs = set(pdb_ids[val_end:])

    train_records = []
    val_records = []
    test_records = []

    for pdb_id, pdb_records in pdb_to_records.items():
        if pdb_id in train_pdbs:
            train_records.extend(pdb_records)
        elif pdb_id in val_pdbs:
            val_records.extend(pdb_records)
        elif pdb_id in test_pdbs:
            test_records.extend(pdb_records)

    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    return train_records, val_records, test_records


def save_records(records: list[dict], output_path: Path) -> None:
    """Save records with a stable column order.

     data/processed/, data/processed_affinity/sequence_only/
     heavy/light classification 
    """

    dataframe = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    dataframe.to_csv(output_path, index=False)


def make_empty_stats() -> dict:
    """Create all counters used in the final summary."""

    return {
        "rows_seen": 0,
        "skipped_invalid_affinity": 0,
        "skipped_missing_chain": 0,
        "skipped_antigen_type": 0,
        "pdb_downloaded": 0,
        "pdb_cache_reused": 0,
        "failed_downloads": 0,
        "failed_pdb_parse": 0,
        "failed_sequence_extractions": 0,
        "skipped_short_sequences": 0,
        "records_built": 0,
    }


def print_target_summary(records: list[dict]) -> None:
    """Print target statistics for beginner sanity checking."""

    if not records:
        print("No records available for target summary.")
        return

    targets = pd.Series([record["neg_log10_affinity"] for record in records], dtype=float)
    print(f"Target neg_log10_affinity min: {targets.min():.4f}")
    print(f"Target neg_log10_affinity max: {targets.max():.4f}")
    print(f"Target neg_log10_affinity mean: {targets.mean():.4f}")
    print(f"Target neg_log10_affinity std: {targets.std():.4f}")


def main() -> None:
    """Main dataset build workflow."""

    args = parse_args()

    summary_path = Path(args.summary_path)
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)

    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary file: {summary_path}")

    pdb_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path, sep="\t", dtype=str, keep_default_na=False)

    missing_columns = set(REQUIRED_COLUMNS) - set(summary.columns)
    if missing_columns:
        raise ValueError(f"Summary file is missing required columns: {missing_columns}")

    if args.max_rows is not None:
        summary = summary.head(args.max_rows)

    stats = make_empty_stats()
    records = []

    print("Building SAbDab affinity regression dataset")
    print(f"Rows to scan: {len(summary)}")
    print(f"Output directory: {output_dir}")
    print()

    for row_index, row in summary.iterrows():
        stats["rows_seen"] += 1

        pdb_id = str(row["pdb"]).strip().upper()
        h_chain = clean_chain_id(row["Hchain"])
        l_chain = clean_chain_id(row["Lchain"])
        antigen_chain_text = str(row["antigen_chain"]).strip()
        antigen_chain_ids = split_chain_ids(antigen_chain_text)

        if stats["rows_seen"] % 25 == 0:
            print(f"Processing row {stats['rows_seen']}/{len(summary)} | PDB: {pdb_id}")

        # affinity , -log10(affinity)
        # missing /  / <=0  regression target
        affinity = pd.to_numeric(row["affinity"], errors="coerce")
        if pd.isna(affinity) or float(affinity) <= 0:
            stats["skipped_invalid_affinity"] += 1
            continue

        #  heavy/light/antigen chain id, PDB 
        if is_missing(h_chain) or is_missing(l_chain) or is_missing(antigen_chain_text):
            stats["skipped_missing_chain"] += 1
            continue

        if not antigen_chain_ids:
            stats["skipped_missing_chain"] += 1
            continue

        # Stage 1  sequence-only , Hapten  antigen
        if not is_valid_antigen_type(row["antigen_type"]):
            stats["skipped_antigen_type"] += 1
            continue

        pdb_path = download_pdb_if_needed(pdb_id, pdb_dir, stats)
        if pdb_path is None:
            continue

        structure = load_structure(pdb_path, pdb_id, stats)
        if structure is None:
            continue

        heavy_sequence = extract_chain_sequence(
            structure, pdb_id, h_chain, args.min_sequence_length, stats
        )
        light_sequence = extract_chain_sequence(
            structure, pdb_id, l_chain, args.min_sequence_length, stats
        )

        antigen_parts = []
        for antigen_chain_id in antigen_chain_ids:
            antigen_part = extract_chain_sequence(
                structure, pdb_id, antigen_chain_id, args.min_sequence_length, stats
            )
            if antigen_part is not None:
                antigen_parts.append(antigen_part)

        if heavy_sequence is None or light_sequence is None or len(antigen_parts) != len(antigen_chain_ids):
            continue

        #  antigen_chain  chain, A|B, antigen sequence 
        antigen_sequence = "".join(antigen_parts)

        # target normalization:
        #  affinity  binding 
        # -log10 , binding ,
        neg_log10_affinity = -math.log10(float(affinity))

        record = {
            "sample_id": f"AFF_{len(records) + 1:06d}",
            "pdb": pdb_id,
            "Hchain": h_chain,
            "Lchain": l_chain,
            "antigen_chain": antigen_chain_text,
            "antigen_type": row["antigen_type"],
            "antigen_name": row["antigen_name"],
            "heavy_sequence": heavy_sequence,
            "light_sequence": light_sequence,
            "antigen_sequence": antigen_sequence,
            "affinity": float(affinity),
            "neg_log10_affinity": neg_log10_affinity,
            "delta_g": row["delta_g"],
            "affinity_method": row["affinity_method"],
            "temperature": row["temperature"],
            "pmid": row["pmid"],
        }
        records.append(record)
        stats["records_built"] += 1

    train_records, val_records, test_records = split_by_pdb(records, args.seed)

    save_records(train_records, output_dir / "train.csv")
    save_records(val_records, output_dir / "val.csv")
    save_records(test_records, output_dir / "test.csv")

    print()
    print("Affinity dataset build summary")
    print(f"Rows scanned: {stats['rows_seen']}")
    print(f"Final dataset size: {len(records)}")
    print(f"Unique PDB count: {pd.Series([record['pdb'] for record in records]).nunique() if records else 0}")
    print(f"Train size: {len(train_records)}")
    print(f"Val size: {len(val_records)}")
    print(f"Test size: {len(test_records)}")
    print(f"PDB files downloaded: {stats['pdb_downloaded']}")
    print(f"PDB files reused from cache: {stats['pdb_cache_reused']}")
    print(f"Skipped invalid affinity: {stats['skipped_invalid_affinity']}")
    print(f"Skipped missing chain: {stats['skipped_missing_chain']}")
    print(f"Skipped antigen type: {stats['skipped_antigen_type']}")
    print(f"Failed downloads: {stats['failed_downloads']}")
    print(f"Failed PDB parses: {stats['failed_pdb_parse']}")
    print(f"Failed sequence extractions: {stats['failed_sequence_extractions']}")
    print(f"Skipped short sequences: {stats['skipped_short_sequences']}")
    print_target_summary(records)


if __name__ == "__main__":
    main()
