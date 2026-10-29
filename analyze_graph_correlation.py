import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_provider.graph_utils import (
    adjacency_from_payload,
    build_gravity_weight_matrix,
    get_population_vector,
    load_region_dataframe,
    normalize_adjacency_matrix,
)


DATASET_SPECS = {
    "California": {
        "root_path": "./dataset/California/",
        "data_path": "COVIDinCalifornia.json",
    },
    "Massachusetts": {
        "root_path": "./dataset/Massachusetts/",
        "data_path": "COVIDinMassachusetts.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute correlation between gravity and mobility networks."
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["California", "Massachusetts"],
        help="Countries / region groups to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/graph_correlation",
        help="Directory for summaries and edge tables.",
    )
    parser.add_argument("--gravity_alpha", type=float, default=1.0)
    parser.add_argument("--gravity_beta", type=float, default=1.0)
    parser.add_argument("--gravity_gamma", type=float, default=2.0)
    return parser.parse_args()


def safe_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    if x.size == 0:
        return {
            "num_edges": 0,
            "pearson_r": np.nan,
            "pearson_p_value": np.nan,
            "pearson_significance_score": np.nan,
            "pearson_significant_0_05": False,
            "pearson_significant_0_01": False,
            "pearson_significant_0_001": False,
            "spearman_r": np.nan,
            "spearman_p_value": np.nan,
            "spearman_significance_score": np.nan,
            "spearman_significant_0_05": False,
            "spearman_significant_0_01": False,
            "spearman_significant_0_001": False,
            "note": "No edges available after masking.",
        }

    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return {
            "num_edges": int(x.size),
            "pearson_r": np.nan,
            "pearson_p_value": np.nan,
            "pearson_significance_score": np.nan,
            "pearson_significant_0_05": False,
            "pearson_significant_0_01": False,
            "pearson_significant_0_001": False,
            "spearman_r": np.nan,
            "spearman_p_value": np.nan,
            "spearman_significance_score": np.nan,
            "spearman_significant_0_05": False,
            "spearman_significant_0_01": False,
            "spearman_significant_0_001": False,
            "note": "Correlation is undefined because one vector is constant.",
        }

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    pearson_score = float(-np.log10(max(float(pearson_p), np.finfo(float).tiny)))
    spearman_score = float(-np.log10(max(float(spearman_p), np.finfo(float).tiny)))
    return {
        "num_edges": int(x.size),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "pearson_significance_score": pearson_score,
        "pearson_significant_0_05": bool(pearson_p < 0.05),
        "pearson_significant_0_01": bool(pearson_p < 0.01),
        "pearson_significant_0_001": bool(pearson_p < 0.001),
        "spearman_r": float(spearman_r),
        "spearman_p_value": float(spearman_p),
        "spearman_significance_score": spearman_score,
        "spearman_significant_0_05": bool(spearman_p < 0.05),
        "spearman_significant_0_01": bool(spearman_p < 0.01),
        "spearman_significant_0_001": bool(spearman_p < 0.001),
        "note": "",
    }


def canonical_country_name(name: str) -> str:
    normalized = name.strip().lower()
    for candidate in DATASET_SPECS:
        if candidate.lower() == normalized:
            return candidate
    raise ValueError(
        f"Unsupported country `{name}`. Choose from: {sorted(DATASET_SPECS)}."
    )


def load_networks(
    country: str,
    gravity_alpha: float,
    gravity_beta: float,
    gravity_gamma: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASET_SPECS[country]
    _, region_df, payload = load_region_dataframe(
        country=country,
        root_path=spec["root_path"],
        data_path=spec["data_path"],
    )
    if payload is None:
        raise ValueError(f"Missing JSON payload for `{country}`.")

    node_names = [str(node) for node in payload.get("node", [])]
    mobility_adj = adjacency_from_payload(payload)
    population = get_population_vector(region_df, payload=payload)
    gravity_adj = build_gravity_weight_matrix(
        region_df,
        population,
        alpha=gravity_alpha,
        beta=gravity_beta,
        gamma=gravity_gamma,
    )
    mobility_norm = normalize_adjacency_matrix(
        mobility_adj, add_self_loops=True, scale_by_max=True
    )
    gravity_norm = normalize_adjacency_matrix(
        gravity_adj, add_self_loops=True, scale_by_max=True
    )
    return node_names, mobility_adj, gravity_adj, mobility_norm, gravity_norm


def build_edge_table(
    node_names: list[str],
    mobility_adj: np.ndarray,
    gravity_adj: np.ndarray,
) -> pd.DataFrame:
    records = []
    n = len(node_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mobility = float(mobility_adj[i, j])
            gravity = float(gravity_adj[i, j])
            records.append(
                {
                    "src_idx": i,
                    "src_name": node_names[i],
                    "dst_idx": j,
                    "dst_name": node_names[j],
                    "mobility_weight": mobility,
                    "gravity_weight": gravity,
                    "mobility_log1p": float(np.log1p(mobility)),
                    "gravity_log1p": float(np.log1p(gravity)),
                    "either_nonzero": bool((mobility != 0.0) or (gravity != 0.0)),
                    "both_nonzero": bool((mobility != 0.0) and (gravity != 0.0)),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_country(
    country: str,
    node_names: list[str],
    mobility_adj: np.ndarray,
    gravity_adj: np.ndarray,
    mobility_norm: np.ndarray,
    gravity_norm: np.ndarray,
) -> dict:
    offdiag_mask = ~np.eye(len(node_names), dtype=bool)
    mobility_offdiag = mobility_adj[offdiag_mask]
    gravity_offdiag = gravity_adj[offdiag_mask]
    mobility_log = np.log1p(mobility_offdiag)
    gravity_log = np.log1p(gravity_offdiag)
    mobility_norm_offdiag = mobility_norm[offdiag_mask]
    gravity_norm_offdiag = gravity_norm[offdiag_mask]
    union_nonzero_mask = (mobility_offdiag != 0.0) | (gravity_offdiag != 0.0)
    both_nonzero_mask = (mobility_offdiag != 0.0) & (gravity_offdiag != 0.0)

    return {
        "country": country,
        "num_nodes": int(len(node_names)),
        "num_directed_edges_offdiag": int(mobility_offdiag.size),
        "mobility_nonzero_edges": int(np.count_nonzero(mobility_offdiag)),
        "gravity_nonzero_edges": int(np.count_nonzero(gravity_offdiag)),
        "mobility_density": float(np.count_nonzero(mobility_offdiag) / mobility_offdiag.size),
        "gravity_density": float(np.count_nonzero(gravity_offdiag) / gravity_offdiag.size),
        "correlation_all_offdiag_raw": safe_correlation(mobility_offdiag, gravity_offdiag),
        "correlation_all_offdiag_log1p": safe_correlation(mobility_log, gravity_log),
        "correlation_all_offdiag_normalized": safe_correlation(
            mobility_norm_offdiag, gravity_norm_offdiag
        ),
        "correlation_union_nonzero_raw": safe_correlation(
            mobility_offdiag[union_nonzero_mask], gravity_offdiag[union_nonzero_mask]
        ),
        "correlation_union_nonzero_log1p": safe_correlation(
            mobility_log[union_nonzero_mask], gravity_log[union_nonzero_mask]
        ),
        "correlation_both_nonzero_raw": safe_correlation(
            mobility_offdiag[both_nonzero_mask], gravity_offdiag[both_nonzero_mask]
        ),
        "correlation_both_nonzero_log1p": safe_correlation(
            mobility_log[both_nonzero_mask], gravity_log[both_nonzero_mask]
        ),
    }


def save_country_outputs(
    output_dir: Path,
    country: str,
    summary: dict,
    edge_table: pd.DataFrame,
) -> None:
    country_dir = output_dir / country.lower()
    country_dir.mkdir(parents=True, exist_ok=True)

    edge_table.to_csv(country_dir / "gravity_vs_mobility_edges.csv", index=False)
    with (country_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (country_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Country: {summary['country']}\n")
        f.write(f"Nodes: {summary['num_nodes']}\n")
        f.write(f"Directed off-diagonal edges: {summary['num_directed_edges_offdiag']}\n")
        f.write(f"Mobility nonzero edges: {summary['mobility_nonzero_edges']}\n")
        f.write(f"Gravity nonzero edges: {summary['gravity_nonzero_edges']}\n")
        f.write(
            "All off-diagonal raw Pearson/Spearman: "
            f"{summary['correlation_all_offdiag_raw']['pearson_r']:.6f} / "
            f"{summary['correlation_all_offdiag_raw']['spearman_r']:.6f}\n"
        )
        f.write(
            "All off-diagonal raw Pearson p-value / score: "
            f"{summary['correlation_all_offdiag_raw']['pearson_p_value']:.6e} / "
            f"{summary['correlation_all_offdiag_raw']['pearson_significance_score']:.6f}\n"
        )
        f.write(
            "All off-diagonal raw Spearman p-value / score: "
            f"{summary['correlation_all_offdiag_raw']['spearman_p_value']:.6e} / "
            f"{summary['correlation_all_offdiag_raw']['spearman_significance_score']:.6f}\n"
        )
        f.write(
            "All off-diagonal log1p Pearson/Spearman: "
            f"{summary['correlation_all_offdiag_log1p']['pearson_r']:.6f} / "
            f"{summary['correlation_all_offdiag_log1p']['spearman_r']:.6f}\n"
        )
        f.write(
            "All off-diagonal log1p Pearson p-value / score: "
            f"{summary['correlation_all_offdiag_log1p']['pearson_p_value']:.6e} / "
            f"{summary['correlation_all_offdiag_log1p']['pearson_significance_score']:.6f}\n"
        )
        f.write(
            "All off-diagonal log1p Spearman p-value / score: "
            f"{summary['correlation_all_offdiag_log1p']['spearman_p_value']:.6e} / "
            f"{summary['correlation_all_offdiag_log1p']['spearman_significance_score']:.6f}\n"
        )
        f.write(
            "All off-diagonal normalized Pearson/Spearman: "
            f"{summary['correlation_all_offdiag_normalized']['pearson_r']:.6f} / "
            f"{summary['correlation_all_offdiag_normalized']['spearman_r']:.6f}\n"
        )
        f.write(
            "All off-diagonal normalized Pearson p-value / score: "
            f"{summary['correlation_all_offdiag_normalized']['pearson_p_value']:.6e} / "
            f"{summary['correlation_all_offdiag_normalized']['pearson_significance_score']:.6f}\n"
        )
        f.write(
            "All off-diagonal normalized Spearman p-value / score: "
            f"{summary['correlation_all_offdiag_normalized']['spearman_p_value']:.6e} / "
            f"{summary['correlation_all_offdiag_normalized']['spearman_significance_score']:.6f}\n"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "gravity_alpha": float(args.gravity_alpha),
        "gravity_beta": float(args.gravity_beta),
        "gravity_gamma": float(args.gravity_gamma),
        "countries": {},
    }

    for name in args.countries:
        country = canonical_country_name(name)
        node_names, mobility_adj, gravity_adj, mobility_norm, gravity_norm = load_networks(
            country=country,
            gravity_alpha=args.gravity_alpha,
            gravity_beta=args.gravity_beta,
            gravity_gamma=args.gravity_gamma,
        )
        edge_table = build_edge_table(node_names, mobility_adj, gravity_adj)
        summary = summarize_country(
            country=country,
            node_names=node_names,
            mobility_adj=mobility_adj,
            gravity_adj=gravity_adj,
            mobility_norm=mobility_norm,
            gravity_norm=gravity_norm,
        )
        save_country_outputs(output_dir, country, summary, edge_table)
        combined["countries"][country] = summary

        raw_corr = summary["correlation_all_offdiag_raw"]
        log_corr = summary["correlation_all_offdiag_log1p"]
        norm_corr = summary["correlation_all_offdiag_normalized"]
        print(
            f"[{country}] raw Pearson={raw_corr['pearson_r']:.6f}, "
            f"p={raw_corr['pearson_p_value']:.6e}, "
            f"score={raw_corr['pearson_significance_score']:.6f}, "
            f"log1p Pearson={log_corr['pearson_r']:.6f}, "
            f"normalized Pearson={norm_corr['pearson_r']:.6f}"
        )
        print(
            f"[{country}] raw Spearman={raw_corr['spearman_r']:.6f}, "
            f"p={raw_corr['spearman_p_value']:.6e}, "
            f"score={raw_corr['spearman_significance_score']:.6f}, "
            f"log1p Spearman={log_corr['spearman_r']:.6f}, "
            f"normalized Spearman={norm_corr['spearman_r']:.6f}"
        )

    with (output_dir / "combined_summary.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
