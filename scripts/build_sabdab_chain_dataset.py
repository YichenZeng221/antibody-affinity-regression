"""Build a heavy-chain vs light-chain dataset from SAbDab.

中文人话说明：
这个脚本只负责“真实数据集构建”，不改模型、不改训练代码。

输入：
    data/raw/sabdab_summary.tsv

输出：
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

每个输出 CSV 保存这些列：
    sequence,label,pdb,chain,chain_type

标签约定：
    label = 1 表示 antibody heavy chain
    label = 0 表示 antibody light chain

为什么先用 max_rows 小规模测试？
SAbDab summary 可能有很多行。每一行都可能要下载 PDB、解析结构、提取 chain，
第一次全量跑会比较久，看起来像“卡住”。所以第一版默认只处理前 100 行。
确认流程没问题后，你可以把 --max_rows 改大，或者传 --max_rows none 跑全量。

运行示例：
    python scripts/build_sabdab_chain_dataset.py --max_rows 100
    python scripts/build_sabdab_chain_dataset.py --max_rows none
"""

from pathlib import Path
import argparse
import random
import urllib.request

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


SUMMARY_PATH = Path("data/raw/sabdab_summary.tsv")
PDB_DIR = Path("data/pdb")
PROCESSED_DIR = Path("data/processed")

DEFAULT_MAX_ROWS = 100
MIN_SEQUENCE_LENGTH = 20
RANDOM_SEED = 42


def parse_max_rows(value: str) -> int | None:
    """把命令行里的 --max_rows 参数转成 int 或 None。

    中文人话说明：
    - --max_rows 100 表示只处理前 100 行
    - --max_rows none 表示不限制行数，处理全量
    """

    text = str(value).strip().lower()
    if text in {"none", "all", "full"}:
        return None
    return int(text)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Build train/val/test CSV files from SAbDab summary TSV."
    )
    parser.add_argument(
        "--max_rows",
        type=parse_max_rows,
        default=DEFAULT_MAX_ROWS,
        help="How many summary rows to process. Use 'none' to process all rows.",
    )
    return parser.parse_args()


def is_missing_chain_id(chain_id: str) -> bool:
    """判断 chain id 是否缺失。

    中文人话说明：
    SAbDab 里 Hchain/Lchain 可能是 NA。
    NA 的意思是：这个 summary row 没告诉我们对应的 chain id。

    如果 chain id 缺失，我们就不知道应该从 PDB 文件里提取哪条链，
    所以必须跳过对应 chain。
    """

    text = str(chain_id).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE"}


def clean_chain_id(chain_id: str) -> str:
    """清理 chain id 前后空格。"""

    return str(chain_id).strip()


def download_pdb_file(pdb_id: str, stats: dict) -> Path | None:
    """下载或复用本地缓存的 PDB 文件。

    中文人话说明：
    为什么要用 pdb + chain id？

    PDB ID 只能定位到一个结构文件，比如 11HK。
    但一个 PDB 文件里通常有很多条 chain：抗原 chain、heavy chain、light chain 等。
    所以必须用：

        pdb id + chain id

    才能准确提取一条 antibody chain sequence。
    """

    pdb_id = pdb_id.upper()
    pdb_path = PDB_DIR / f"{pdb_id}.pdb"

    if pdb_path.exists():
        print(f"  using cached PDB: {pdb_id}")
        stats["pdb_cache_reused"] += 1
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  downloading PDB: {pdb_id} from {url}")

    try:
        urllib.request.urlretrieve(url, pdb_path)
        stats["pdb_downloaded"] += 1
        return pdb_path
    except Exception as error:
        print(f"  WARNING: failed to download PDB {pdb_id}: {error}")
        stats["failed_downloads"] += 1
        return None


def extract_chain_sequence(
    pdb_path: Path,
    pdb_id: str,
    chain_id: str,
    chain_name: str,
    stats: dict,
) -> str | None:
    """从 PDB 文件里提取指定 chain 的 amino acid sequence。

    中文人话说明：
    PDB 里的残基通常是三字母代码，比如 ALA、GLY、TYR。
    ESM-2 需要一字母氨基酸序列，所以这里会转成 A、G、Y。

    chain_name 只是为了打印更清楚：
    - "heavy" 表示 Hchain
    - "light" 表示 Lchain
    """

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as error:
        print(f"  WARNING: failed to parse PDB file {pdb_path}: {error}")
        stats["failed_sequence_extractions"] += 1
        return None

    model = structure[0]

    if chain_id not in model:
        print(f"  WARNING: extraction failed, PDB {pdb_id} has no chain {chain_id}")
        stats["failed_sequence_extractions"] += 1
        return None

    amino_acids = []
    for residue in model[chain_id]:
        if not is_aa(residue, standard=True):
            continue
        amino_acids.append(seq1(residue.get_resname()))

    sequence = "".join(amino_acids)

    if sequence == "":
        print(f"  WARNING: extraction failed, PDB {pdb_id} chain {chain_id} is empty")
        stats["failed_sequence_extractions"] += 1
        return None

    if len(sequence) < MIN_SEQUENCE_LENGTH:
        print(
            f"  skipped short sequence: {chain_name} chain {chain_id}, "
            f"length={len(sequence)}"
        )
        stats["skipped_short_sequences"] += 1
        return None

    print(f"  extracted {chain_name} sequence: chain {chain_id}, length={len(sequence)}")
    return sequence


def remove_duplicate_sequences(records: list[dict], stats: dict) -> list[dict]:
    """删除重复 sequence。

    中文人话说明：
    为什么要删除重复 sequence？

    如果同一条 sequence 同时出现在 train 和 test，
    模型可能只是记住了这条序列，而不是真的学会 heavy/light 的区别。
    去重可以让后面的评估更干净一点。
    """

    seen_sequences = set()
    unique_records = []

    for record in records:
        sequence = record["sequence"]
        if sequence in seen_sequences:
            stats["duplicates_removed"] += 1
            continue

        seen_sequences.add(sequence)
        unique_records.append(record)

    return unique_records


def split_by_pdb(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """按 PDB ID 切分 train/val/test。

    中文人话说明：
    为什么 PDB-level split 比 random sequence split 更严格？

    random sequence split 是把每条 sequence 随机分到 train/val/test。
    这样可能会出现：

        同一个 PDB 的 heavy chain 在 train
        同一个 PDB 的 light chain 在 test

    这会让 test set 和 train set 太接近，评估可能虚高。

    PDB-level split 的规则是：

        同一个 pdb 的所有 chain 必须只出现在一个 split

    这样 test set 更像“模型没见过的新结构来源”，更严格。

    注意：
    这里是按 PDB 切分，不是按 label 切分。
    所以 label 比例可能不会完美 50/50。
    后面用 check_processed_dataset.py 检查 label distribution。
    """

    random.seed(RANDOM_SEED)

    pdb_to_records = {}
    for record in records:
        pdb_id = record["pdb"]
        if pdb_id not in pdb_to_records:
            pdb_to_records[pdb_id] = []
        pdb_to_records[pdb_id].append(record)

    pdb_ids = list(pdb_to_records.keys())
    random.shuffle(pdb_ids)

    total_pdbs = len(pdb_ids)
    train_end = int(total_pdbs * 0.8)
    val_end = int(total_pdbs * 0.9)

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
    """保存 CSV。

    中文人话说明：
    训练代码只需要 sequence,label。
    但是我们额外保存 pdb、chain、chain_type，方便检查 data leakage。
    """

    dataframe = pd.DataFrame(
        records,
        columns=["sequence", "label", "pdb", "chain", "chain_type"],
    )
    dataframe.to_csv(output_path, index=False)


def make_empty_stats() -> dict:
    """创建统计计数器。"""

    return {
        "pdb_downloaded": 0,
        "pdb_cache_reused": 0,
        "heavy_extracted": 0,
        "light_extracted": 0,
        "failed_downloads": 0,
        "failed_sequence_extractions": 0,
        "skipped_same_chain": 0,
        "skipped_na_chains": 0,
        "skipped_short_sequences": 0,
        "duplicates_removed": 0,
    }


def main() -> None:
    """脚本主函数：从 SAbDab summary 构建真实 heavy/light 数据集。"""

    args = parse_args()
    max_rows = args.max_rows

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find summary file: {SUMMARY_PATH}")

    PDB_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(
        SUMMARY_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {"pdb", "Hchain", "Lchain"}
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"Summary file is missing columns: {missing_columns}")

    total_summary_rows = len(summary)

    if max_rows is None:
        rows_to_process = summary
    else:
        rows_to_process = summary.head(max_rows)

    rows_processed = len(rows_to_process)
    stats = make_empty_stats()
    records = []

    print("Building SAbDab heavy-chain vs light-chain dataset")
    print(f"Total summary rows read: {total_summary_rows}")
    print(f"Rows to process this run: {rows_processed}")
    print()

    for processed_index, (_, row) in enumerate(rows_to_process.iterrows(), start=1):
        pdb_id = str(row["pdb"]).strip().upper()
        h_chain = clean_chain_id(row["Hchain"])
        l_chain = clean_chain_id(row["Lchain"])

        print(
            f"Processing {processed_index}/{rows_processed} | "
            f"PDB: {pdb_id} | Hchain: {h_chain} | Lchain: {l_chain}"
        )

        if pdb_id == "":
            print("  skipped row because PDB ID is empty")
            continue

        h_missing = is_missing_chain_id(h_chain)
        l_missing = is_missing_chain_id(l_chain)

        if not h_missing and not l_missing and h_chain == l_chain:
            print("  skipped because Hchain == Lchain")
            stats["skipped_same_chain"] += 1
            continue

        if h_missing and l_missing:
            print("  skipped because both Hchain and Lchain are NA")
            stats["skipped_na_chains"] += 1
            continue

        if h_missing or l_missing:
            print("  skipped missing side because Hchain or Lchain is NA")
            stats["skipped_na_chains"] += 1

        pdb_path = download_pdb_file(pdb_id, stats)
        if pdb_path is None:
            continue

        if not h_missing:
            sequence = extract_chain_sequence(pdb_path, pdb_id, h_chain, "heavy", stats)
            if sequence is not None:
                records.append(
                    {
                        "sequence": sequence,
                        "label": 1,
                        "pdb": pdb_id,
                        "chain": h_chain,
                        "chain_type": "heavy",
                    }
                )
                stats["heavy_extracted"] += 1

        if not l_missing:
            sequence = extract_chain_sequence(pdb_path, pdb_id, l_chain, "light", stats)
            if sequence is not None:
                records.append(
                    {
                        "sequence": sequence,
                        "label": 0,
                        "pdb": pdb_id,
                        "chain": l_chain,
                        "chain_type": "light",
                    }
                )
                stats["light_extracted"] += 1

    unique_records = remove_duplicate_sequences(records, stats)
    train_records, val_records, test_records = split_by_pdb(unique_records)

    save_records(train_records, PROCESSED_DIR / "train.csv")
    save_records(val_records, PROCESSED_DIR / "val.csv")
    save_records(test_records, PROCESSED_DIR / "test.csv")

    print()
    print("SAbDab dataset build summary")
    print(f"Total summary rows read: {total_summary_rows}")
    print(f"Rows processed: {rows_processed}")
    print(f"PDB files downloaded: {stats['pdb_downloaded']}")
    print(f"PDB files reused from cache: {stats['pdb_cache_reused']}")
    print(f"Heavy chains extracted: {stats['heavy_extracted']}")
    print(f"Light chains extracted: {stats['light_extracted']}")
    print(f"Failed downloads: {stats['failed_downloads']}")
    print(f"Failed sequence extractions: {stats['failed_sequence_extractions']}")
    print(f"Skipped Hchain == Lchain: {stats['skipped_same_chain']}")
    print(f"Skipped Hchain or Lchain is NA: {stats['skipped_na_chains']}")
    print(f"Skipped short sequences: {stats['skipped_short_sequences']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Final dataset size: {len(unique_records)}")
    print(f"Train size: {len(train_records)}")
    print(f"Val size: {len(val_records)}")
    print(f"Test size: {len(test_records)}")


if __name__ == "__main__":
    main()
