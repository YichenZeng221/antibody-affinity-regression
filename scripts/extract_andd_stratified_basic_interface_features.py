"""Extract a conservative basic interface geometry pilot for ANDD antibody v2.

本脚本只处理 contact audit 中已经证明 chain mapping 不歧义的样本。
它读取外部 SAbDab raw PDB 文件，计算最基础的抗体-抗原界面几何特征，
再与已经存在的 test predictions 按 sample_id 合并，用于误差分析。

重要边界：
- 不训练模型，不修改原始 dataset。
- 不处理 ambiguous chain mappings，也不猜 chain ID。
- 暂不计算 CDR-specific contact；那需要先验证 IMGT CDR residue 到结构 residue 的对应关系。

这里将 contact count 定义为 residue-pair contact count：
只要一个 antibody residue 与一个 antigen residue 存在任一对非氢原子距离不大于阈值，
这个 residue pair 就计为一个 contact。这样比 atom-pair count 更容易解释。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa


DATA_DIR = Path("data/processed_affinity/expanded_affinity_antibody_v2_stratified")
AVAILABILITY_PATH = Path(
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/contact_feature_availability.csv"
)
ARCHIVE_RAW_DIR = Path("/Users/yichenzeng/Downloads/all_structures/raw")
OUTPUT_DIR = Path("outputs/andd_antibody_v2_stratified/contact_feature_audit")
FEATURE_PATH = OUTPUT_DIR / "basic_interface_features.csv"
REPORT_PATH = OUTPUT_DIR / "basic_interface_feature_report.md"
FIGURE_PATH = Path("outputs/final_reports/figures/basic_interface_feature_correlations.png")

PREDICTION_FILES = {
    "unweighted_cross_attention": Path(
        "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/test_predictions.csv"
    ),
    "tailaware_w2_best_val_tail_mae": Path(
        "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/"
        "tailaware_w2_test_predictions_best_val_tail_mae.csv"
    ),
}

TARGET_COLUMN = "neg_log10_affinity_candidate"
CUTOFFS = (4.0, 5.0, 8.0)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "na", "none", "\\"} else text


def parse_chain_ids(value: object) -> list[str]:
    """Audit 已提供 resolved chain；这里保留多 antigen chains 的兼容性。"""
    text = clean_text(value)
    for separator in ("|", ",", ";", "/"):
        text = text.replace(separator, " ")
    return [token for token in text.split() if token]


def raw_pdb_index() -> dict[str, Path]:
    if not ARCHIVE_RAW_DIR.exists():
        raise FileNotFoundError(f"External SAbDab raw structure directory not found: {ARCHIVE_RAW_DIR}")
    return {path.stem.upper(): path for path in ARCHIVE_RAW_DIR.glob("*.pdb")}


def heavy_atoms_by_residue(chain) -> dict[tuple[str, int, str], list[np.ndarray]]:
    """只保留 amino-acid residue 的非氢原子坐标。"""
    residues: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for residue in chain:
        if not is_aa(residue, standard=False):
            continue
        residue_id = (chain.id, int(residue.id[1]), str(residue.id[2]).strip())
        coords = [
            atom.coord.astype(float)
            for atom in residue.get_atoms()
            if str(getattr(atom, "element", "")).strip().upper() != "H"
        ]
        if coords:
            residues[residue_id] = coords
    return residues


def combine_residues(model, chain_ids: list[str]) -> dict[tuple[str, int, str], list[np.ndarray]]:
    residues: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for chain_id in chain_ids:
        if chain_id not in model:
            raise KeyError(f"chain_not_in_structure:{chain_id}")
        residues.update(heavy_atoms_by_residue(model[chain_id]))
    return residues


def minimum_atom_distance(
    ab_residues: dict[tuple[str, int, str], list[np.ndarray]],
    ag_residues: dict[tuple[str, int, str], list[np.ndarray]],
) -> float:
    ab_coords = np.asarray([coord for coords in ab_residues.values() for coord in coords])
    ag_coords = np.asarray([coord for coords in ag_residues.values() for coord in coords])
    if ab_coords.size == 0 or ag_coords.size == 0:
        return float("nan")
    minimum = float("inf")
    # 分块避免对较大 antigen 一次分配过大的三维矩阵。
    for start in range(0, len(ab_coords), 256):
        chunk = ab_coords[start : start + 256]
        distances = np.linalg.norm(chunk[:, None, :] - ag_coords[None, :, :], axis=2)
        minimum = min(minimum, float(distances.min()))
    return minimum


def residue_contacts(
    ab_residues: dict[tuple[str, int, str], list[np.ndarray]],
    ag_residues: dict[tuple[str, int, str], list[np.ndarray]],
    cutoff: float,
) -> set[tuple[tuple[str, int, str], tuple[str, int, str]]]:
    """返回 cutoff 范围内接触的 residue-pair set。

    使用空间索引查询附近原子，而不是逐个 residue pair 扫描，面对较长 antigen
    时会快很多，contact 的定义保持不变。
    """
    contacts: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    antigen_atoms = []
    antigen_atom_residue: dict[int, tuple[str, int, str]] = {}
    for ag_id, ag_coords_list in ag_residues.items():
        for coord in ag_coords_list:
            # NeighborSearch 接受具有 get_coord 方法的对象；临时包装坐标即可。
            atom = _CoordinateAtom(coord)
            antigen_atoms.append(atom)
            antigen_atom_residue[id(atom)] = ag_id
    if not antigen_atoms:
        return contacts
    search = NeighborSearch(antigen_atoms)
    for ab_id, ab_coords_list in ab_residues.items():
        for coord in ab_coords_list:
            for antigen_atom in search.search(coord, cutoff, level="A"):
                contacts.add((ab_id, antigen_atom_residue[id(antigen_atom)]))
    return contacts


class _CoordinateAtom:
    """供 Bio.PDB.NeighborSearch 使用的最小坐标对象。"""

    def __init__(self, coord: np.ndarray) -> None:
        self.coord = coord

    def get_coord(self) -> np.ndarray:
        return self.coord


def extract_geometry(
    pdb_path: Path,
    heavy_chains: list[str],
    light_chains: list[str],
    antigen_chains: list[str],
) -> dict[str, float]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    heavy = combine_residues(model, heavy_chains)
    light = combine_residues(model, light_chains)
    antigen = combine_residues(model, antigen_chains)
    antibody = {**heavy, **light}
    if not antibody or not antigen:
        raise ValueError("empty_antibody_or_antigen_amino_acid_residues")

    contacts = {cutoff: residue_contacts(antibody, antigen, cutoff) for cutoff in CUTOFFS}
    contacts_5a = contacts[5.0]
    return {
        "min_ab_ag_distance": minimum_atom_distance(antibody, antigen),
        "contact_count_4A": float(len(contacts[4.0])),
        "contact_count_5A": float(len(contacts_5a)),
        "contact_count_8A": float(len(contacts[8.0])),
        "antibody_interface_residue_count_5A": float(
            len({ab_id for ab_id, _ in contacts_5a})
        ),
        "antigen_interface_residue_count_5A": float(
            len({ag_id for _, ag_id in contacts_5a})
        ),
        "heavy_interface_residue_count_5A": float(
            len({ab_id for ab_id, _ in contacts_5a if ab_id[0] in set(heavy_chains)})
        ),
        "light_interface_residue_count_5A": float(
            len({ab_id for ab_id, _ in contacts_5a if ab_id[0] in set(light_chains)})
        ),
    }


def load_pilot_rows() -> pd.DataFrame:
    availability = pd.read_csv(AVAILABILITY_PATH)
    if "chain_mapping_status" in availability.columns:
        mask = availability["chain_mapping_status"].astype(str).str.lower().eq("unambiguous")
    else:
        mask = availability["basic_interface_features_ready_for_extraction"].fillna(False).astype(bool)
    pilot = availability.loc[mask].copy()
    return pilot


def add_dataset_columns(pilot: pd.DataFrame) -> pd.DataFrame:
    full = pd.concat(
        [pd.read_csv(DATA_DIR / f"{split}.csv").assign(split=split) for split in ("train", "val", "test")],
        ignore_index=True,
    )
    extra = [
        "sample_id",
        "affinity_kd_m",
        TARGET_COLUMN,
        "heavy_len",
        "light_len",
        "antigen_len",
        "HCDR3",
        "LCDR3",
    ]
    extra = [column for column in extra if column in full.columns]
    base_columns = [column for column in extra if column != "sample_id"]
    return pilot.merge(full[["sample_id", *base_columns]], on="sample_id", how="left", suffixes=("", "_dataset"))


def extract_features(pilot: pd.DataFrame) -> pd.DataFrame:
    pdb_files = raw_pdb_index()
    cache: dict[tuple[str, str, str, str], dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for _, row in pilot.iterrows():
        pdb_id = clean_text(row["pdb_norm"]).upper()
        heavy = clean_text(row["resolved_Hchain"])
        light = clean_text(row["resolved_Lchain"])
        antigen = clean_text(row["resolved_antigen_chain"])
        key = (pdb_id, heavy, light, antigen)
        if key not in cache:
            try:
                path = pdb_files.get(pdb_id)
                if path is None:
                    raise FileNotFoundError("raw_pdb_file_missing")
                geometry = extract_geometry(
                    path,
                    parse_chain_ids(heavy),
                    parse_chain_ids(light),
                    parse_chain_ids(antigen),
                )
                cache[key] = {
                    "geometry_extraction_status": "success",
                    "geometry_extraction_error": "",
                    **geometry,
                }
            except Exception as error:  # 单条结构异常不能中断全部 pilot。
                cache[key] = {
                    "geometry_extraction_status": "failed",
                    "geometry_extraction_error": str(error),
                    **{name: float("nan") for name in geometry_feature_names()},
                }
        records.append({**row.to_dict(), **cache[key]})
    return pd.DataFrame(records)


def geometry_feature_names() -> list[str]:
    return [
        "min_ab_ag_distance",
        "contact_count_4A",
        "contact_count_5A",
        "contact_count_8A",
        "antibody_interface_residue_count_5A",
        "antigen_interface_residue_count_5A",
        "heavy_interface_residue_count_5A",
        "light_interface_residue_count_5A",
    ]


def merge_predictions(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    merged = features.copy()
    missing: list[str] = []
    for model, path in PREDICTION_FILES.items():
        if not path.exists():
            missing.append(str(path))
            continue
        prediction = pd.read_csv(path)
        wanted = [
            column for column in (
                "sample_id",
                "true_neg_log10_affinity",
                "predicted_neg_log10_affinity",
                "error",
                "absolute_error",
            )
            if column in prediction.columns
        ]
        prediction = prediction[wanted].copy()
        rename = {
            column: f"{model}_{column}"
            for column in wanted
            if column != "sample_id"
        }
        prediction.rename(columns=rename, inplace=True)
        merged = merged.merge(prediction, on="sample_id", how="left")
    return merged, missing


def safe_corr(frame: pd.DataFrame, x: str, y: str, method: str) -> float:
    valid = frame[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def correlation_table(features: pd.DataFrame) -> pd.DataFrame:
    success = features[features["geometry_extraction_status"] == "success"].copy()
    rows: list[dict[str, object]] = []
    for feature in geometry_feature_names():
        for outcome in [TARGET_COLUMN]:
            rows.append(
                {
                    "subset": "all_pilot_success",
                    "model": "target",
                    "feature": feature,
                    "outcome": outcome,
                    "n": int(success[[feature, outcome]].dropna().shape[0]),
                    "pearson": safe_corr(success, feature, outcome, "pearson"),
                    "spearman": safe_corr(success, feature, outcome, "spearman"),
                }
            )
        for model in PREDICTION_FILES:
            for outcome_suffix in ("absolute_error", "error"):
                outcome = f"{model}_{outcome_suffix}"
                if outcome not in success.columns:
                    continue
                rows.append(
                    {
                        "subset": "prediction_matched_pilot_success",
                        "model": model,
                        "feature": feature,
                        "outcome": outcome_suffix,
                        "n": int(success[[feature, outcome]].dropna().shape[0]),
                        "pearson": safe_corr(success, feature, outcome, "pearson"),
                        "spearman": safe_corr(success, feature, outcome, "spearman"),
                    }
                )
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def tail_summary(features: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    success = features[features["geometry_extraction_status"] == "success"].copy()
    success["target_tail"] = np.select(
        [success[TARGET_COLUMN] <= lower, success[TARGET_COLUMN] >= upper],
        ["below_train_P10", "above_train_P90"],
        default="middle_P10_to_P90",
    )
    summaries: list[dict[str, object]] = []
    for label, group in success.groupby("target_tail"):
        summaries.append(
            {
                "target_tail": label,
                "n": len(group),
                "target_mean": group[TARGET_COLUMN].mean(),
                "min_distance_mean": group["min_ab_ag_distance"].mean(),
                "contact_count_5A_mean": group["contact_count_5A"].mean(),
                "antibody_interface_residue_count_5A_mean": group[
                    "antibody_interface_residue_count_5A"
                ].mean(),
                "antigen_interface_residue_count_5A_mean": group[
                    "antigen_interface_residue_count_5A"
                ].mean(),
            }
        )
    return pd.DataFrame(summaries)


def create_figure(features: pd.DataFrame) -> None:
    success = features[features["geometry_extraction_status"] == "success"]
    w2_abs = "tailaware_w2_best_val_tail_mae_absolute_error"
    w2_resid = "tailaware_w2_best_val_tail_mae_error"
    test = success.dropna(subset=[w2_abs, w2_resid]) if w2_abs in success else success.iloc[0:0]

    plt.rcParams["font.family"] = "DejaVu Sans"
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].scatter(success["contact_count_5A"], success[TARGET_COLUMN], s=18, alpha=0.55)
    axes[0, 0].set(title=f"Target vs contact count at 5 A (n={len(success)})",
                   xlabel="Residue-pair contact count (<=5 A)", ylabel="neg_log10_affinity")
    axes[0, 1].scatter(success["min_ab_ag_distance"], success[TARGET_COLUMN], s=18, alpha=0.55)
    axes[0, 1].set(title="Target vs minimum interface distance",
                   xlabel="Minimum antibody-antigen distance (A)", ylabel="neg_log10_affinity")
    axes[1, 0].scatter(test["contact_count_5A"], test[w2_abs], s=30, alpha=0.72, color="#d55e00")
    axes[1, 0].set(title=f"Tail-aware w2 absolute error vs contacts (test n={len(test)})",
                   xlabel="Residue-pair contact count (<=5 A)", ylabel="Absolute error")
    axes[1, 1].scatter(test["min_ab_ag_distance"], test[w2_resid], s=30, alpha=0.72, color="#0072b2")
    axes[1, 1].axhline(0, linestyle="--", linewidth=1, color="black")
    axes[1, 1].set(title="Tail-aware w2 residual vs min distance",
                   xlabel="Minimum antibody-antigen distance (A)", ylabel="Prediction - true")
    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    figure.suptitle("ANDD stratified basic interface geometry pilot", fontsize=14)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_report(
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    tails: pd.DataFrame,
    lower: float,
    upper: float,
    missing_predictions: list[str],
) -> None:
    success = features[features["geometry_extraction_status"] == "success"]
    failures = features[features["geometry_extraction_status"] != "success"]
    failure_counts = failures["geometry_extraction_error"].value_counts().rename_axis("reason").reset_index(name="rows")
    target_rows = correlations[
        (correlations["model"] == "target")
        & (correlations["feature"].isin(["contact_count_5A", "min_ab_ag_distance"]))
    ].copy()
    error_rows = correlations[
        (correlations["model"].isin(PREDICTION_FILES))
        & (correlations["feature"].isin(["contact_count_5A", "min_ab_ag_distance"]))
    ].copy()
    target_contact_spearman = float(
        target_rows.loc[target_rows["feature"] == "contact_count_5A", "spearman"].iloc[0]
    )
    target_distance_spearman = float(
        target_rows.loc[target_rows["feature"] == "min_ab_ag_distance", "spearman"].iloc[0]
    )
    very_short_distance_rows = int((success["min_ab_ag_distance"] < 1.0).sum())
    for frame in (target_rows, error_rows):
        for column in ("pearson", "spearman"):
            frame[column] = frame[column].map(lambda value: fmt(value))

    lines = [
        "# ANDD Antibody v2 Stratified Basic Interface Geometry Feature Pilot",
        "",
        "## Scope",
        "",
        "- 本 pilot 只处理 contact availability audit 中 `basic_interface_features_ready_for_extraction=True` "
        "的无歧义 chain-mapping rows。",
        "- 没有处理 ambiguous chain mappings，没有猜测 chain ID，没有训练模型或修改 dataset。",
        "- 结构来源：只读访问 `/Users/yichenzeng/Downloads/all_structures/raw/`。",
        "- `contact_count_*` 的定义是 antibody-antigen **residue pair** contact count："
        "任意一对非氢原子距离小于等于 cutoff，即计为一个接触 residue pair。",
        "- 本轮暂不计算 CDR-specific contacts，因为尚未完成 IMGT CDR residue 到结构 residue 的映射验证。",
        "",
        "## Extraction Result",
        "",
        f"- Pilot-eligible rows from audit: **{len(features)}**.",
        f"- Successfully extracted basic geometry features: **{len(success)} / {len(features)}**.",
        f"- Failed rows: **{len(failures)} / {len(features)}**.",
        f"- Successful rows by split: `{success.groupby('split').size().to_dict()}`.",
        "",
        "### Failure Reasons",
        "",
    ]
    if failure_counts.empty:
        lines.append("- None. All eligible rows were parsed successfully.")
    else:
        lines.extend(markdown_table(failure_counts, ["reason", "rows"]))
    lines.extend(
        [
            "",
            "## Extracted Features",
            "",
            "- `min_ab_ag_distance`: antibody heavy/light chains 到 antigen chain(s) 的最小非氢原子距离。",
            "- `contact_count_4A`, `contact_count_5A`, `contact_count_8A`: 不同 cutoff 下的 interface residue-pair 数。",
            "- `antibody_interface_residue_count_5A`, `antigen_interface_residue_count_5A`: 5 A 内涉及的两侧 residue 数。",
            "- `heavy_interface_residue_count_5A`, `light_interface_residue_count_5A`: 5 A 内分别来自 heavy/light chain 的 interface residue 数。",
            "",
            "## Feature vs Target Affinity",
            "",
        ]
    )
    lines.extend(markdown_table(target_rows, ["feature", "n", "pearson", "spearman"]))
    lines.extend(
        [
            "",
            "相关性是探索性诊断，不代表因果关系；interface 大小和 affinity 也可能受 antigen 类型、"
            "assay noise、结构构象和 label source 共同影响。",
            f"- 初步观察：`contact_count_5A` 与 target 只有弱关系（Spearman = "
            f"{target_contact_spearman:.3f}），`min_ab_ag_distance` 基本无单变量关系 "
            f"（Spearman = {target_distance_spearman:.3f}）。",
            f"- Geometry QC 注意：有 **{very_short_distance_rows}** row 的最小距离 `< 1.0 A`；"
            "该异常短距离应在进入建模前核查结构 alternate locations、链选择或坐标质量。",
            "",
            "## Feature vs Existing Prediction Error",
            "",
        ]
    )
    if error_rows.empty:
        lines.append("- No matching prediction/residual files were available.")
    else:
        lines.extend(markdown_table(error_rows, ["model", "feature", "outcome", "n", "pearson", "spearman"]))
    if missing_predictions:
        lines.extend(["", "Missing optional prediction files:"] + [f"- `{path}`" for path in missing_predictions])
    lines.extend(
        [
            "",
            "## Tail Contact Pattern Audit",
            "",
            f"- Tail thresholds are defined from train targets only: P10 = **{lower:.4f}**, "
            f"P90 = **{upper:.4f}**.",
        ]
    )
    tail_display = tails.copy()
    for column in tail_display.columns:
        if column != "target_tail" and column != "n":
            tail_display[column] = tail_display[column].map(lambda value: fmt(value))
    lines.extend(
        markdown_table(
            tail_display,
            [
                "target_tail",
                "n",
                "target_mean",
                "min_distance_mean",
                "contact_count_5A_mean",
                "antibody_interface_residue_count_5A_mean",
                "antigen_interface_residue_count_5A_mean",
            ],
        )
    )
    high_contact = float(
        tails.loc[tails["target_tail"] == "above_train_P90", "contact_count_5A_mean"].iloc[0]
    )
    low_contact = float(
        tails.loc[tails["target_tail"] == "below_train_P10", "contact_count_5A_mean"].iloc[0]
    )
    lines.extend(
        [
            "",
            f"- 在这批安全子集里，high-tail 的平均 `contact_count_5A` 为 **{high_contact:.3f}**，"
            f"low-tail 为 **{low_contact:.3f}**；这是值得验证的模式，但尚不足以单独解释 affinity tail。",
            "- 特别是 error correlation 仅基于 58 条 test pilot rows，因此不能据此声称 contact "
            "features 已经解决 regression-to-the-mean。",
            "",
            "## Should These Features Enter a Next Model?",
            "",
            "- 这些 features 值得进入下一步 **分析/小型增量 baseline**，因为它们提供了 sequence-only "
            "模型没有看到的真实界面几何信息，并且能够按 `sample_id` 接到已有 residual。",
            "- 但这仍是 pilot 子集：只有无歧义 chain mapping 的样本被纳入。若直接训练，必须清楚说明 "
            "subset selection 会改变 benchmark，并先验证 geometry-error relationship 是否稳定。",
            "- 推荐下一步先在当前提取成功的 test pilot rows 上解读相关性，再决定是否为 train/val/test "
            "全部可解析 subset 建独立 contact-feature benchmark。",
            "",
            "## CDR-specific Contacts: Still Missing",
            "",
            "- CSV 中已有 AbNumber + IMGT 的 CDR sequences；结构中也有 IMGT 文件可供核查。",
            "- 仍需验证 heavy/light sequence 与结构 residue numbering/alignment 的一一对应，特别是 "
            "insertion、missing residues、多模型/多复合物链的处理。",
            "- 在该验证通过前，不生成 `CDR-antigen contact count`、`HCDR3 contact fraction` 或 "
            "`LCDR3 contact fraction`，以免把错误链或错误 residue 范围作为生物学信号。",
            "",
            "## Outputs",
            "",
            f"- Features: `{FEATURE_PATH}`",
            f"- Figure: `{FIGURE_PATH}`",
            f"- Report: `{REPORT_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract basic interface geometry feature pilot.")
    parser.add_argument(
        "--reuse-extracted-features",
        action="store_true",
        help="Reuse this pilot's existing feature CSV and only regenerate report/figure.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse_extracted_features:
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"Cannot reuse features because this pilot output does not exist: {FEATURE_PATH}"
            )
        merged = pd.read_csv(FEATURE_PATH)
        missing_predictions = [
            str(path) for path in PREDICTION_FILES.values() if not path.exists()
        ]
    else:
        pilot = add_dataset_columns(load_pilot_rows())
        extracted = extract_features(pilot)
        merged, missing_predictions = merge_predictions(extracted)
        merged.to_csv(FEATURE_PATH, index=False)

    train = pd.read_csv(DATA_DIR / "train.csv")
    lower = float(train[TARGET_COLUMN].quantile(0.10))
    upper = float(train[TARGET_COLUMN].quantile(0.90))
    correlations = correlation_table(merged)
    tails = tail_summary(merged, lower, upper)
    create_figure(merged)
    write_report(merged, correlations, tails, lower, upper, missing_predictions)

    succeeded = int((merged["geometry_extraction_status"] == "success").sum())
    print("Basic interface geometry feature pilot complete.")
    print(f"Pilot-eligible unambiguous rows: {len(merged)}")
    print(f"Successful extraction rows: {succeeded}")
    print(f"Failed extraction rows: {len(merged) - succeeded}")
    print(f"Features: {FEATURE_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
