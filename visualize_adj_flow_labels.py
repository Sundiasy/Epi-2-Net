"""
Visualize adjacency matrix, mobility flow graph, and label time series.

Examples:
  python visualize_adj_flow_labels.py
  python visualize_adj_flow_labels.py --dataset California
  python visualize_adj_flow_labels.py --root_path ./dataset/California --data_path COVIDinCalifornia.json
  python visualize_adj_flow_labels.py --dataset Italy --graph_csv ./dataset/Italy/graphs/IT_2020-03-13.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "networkx is required for the mobility graph. Please install it first."
    ) from exc


def infer_data_path(root_path: Path) -> Optional[str]:
    for pattern in ("*.json", "*_labels.csv"):
        matches = sorted(root_path.glob(pattern))
        if matches:
            return matches[0].name
    return None


def infer_adj_path(root_path: Path) -> Optional[Path]:
    matches = sorted(root_path.glob("*_adj.npy"))
    return matches[0] if matches else None


def load_json_dataset(path: Path) -> Tuple[pd.DataFrame, dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    nodes = payload.get("node", [])
    times = payload.get("time", [])
    cases = np.asarray(payload.get("case", []), dtype=float)
    if cases.shape != (len(times), len(nodes)):
        raise ValueError(
            f"Case shape {cases.shape} does not match time/node lengths."
        )

    labels = pd.DataFrame(cases, index=times, columns=nodes)
    return labels, payload


def load_csv_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    if "name" not in labels.columns:
        raise ValueError('CSV labels must include a "name" column.')
    labels = labels.set_index("name")
    labels = labels.T
    return labels


def parse_label_nodes(
    labels: pd.DataFrame,
    label_nodes: Optional[str],
    num_label_nodes: int,
) -> list:
    if label_nodes:
        selected = []
        for token in label_nodes.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                idx = int(token)
                if idx < 0 or idx >= labels.shape[1]:
                    raise ValueError(f"Label node index out of range: {idx}")
                selected.append(labels.columns[idx])
            else:
                if token not in labels.columns:
                    raise ValueError(f"Label node not found: {token}")
                selected.append(token)
        return selected

    # default: top nodes by average label value
    scores = labels.mean(axis=0)
    top = scores.sort_values(ascending=False).head(num_label_nodes)
    return list(top.index)


def build_flow_graph_from_json(payload: dict) -> Tuple[nx.DiGraph, Optional[np.ndarray]]:
    nodes = payload.get("node", [])
    edges = payload.get("G", [])
    flows = payload.get("flow", [])
    if not edges or not flows:
        raise ValueError("No flow edges found in JSON payload.")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (src, dst), w in zip(edges, flows):
        src_name = nodes[int(src)]
        dst_name = nodes[int(dst)]
        G.add_edge(src_name, dst_name, weight=float(w))

    population = payload.get("population")
    pop_arr = None
    if population and len(population) == len(nodes):
        pop_arr = np.asarray(population, dtype=float)
    return G, pop_arr


def build_flow_graph_from_csv(path: Path) -> nx.DiGraph:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        raise ValueError("Flow CSV must have at least 3 columns: src, dst, weight.")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = row[0]
        dst = row[1]
        weight = float(row[2])
        G.add_edge(src, dst, weight=weight)
    return G


def select_top_edges(G: nx.DiGraph, top_k: int) -> nx.DiGraph:
    if top_k <= 0:
        return G
    edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)
    edges = edges[: min(top_k, len(edges))]
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(edges)
    return H


def plot_adj_heatmap(adj: np.ndarray, out_path: Path) -> None:
    n = adj.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    show_ticks = n <= 30
    sns.heatmap(
        adj,
        cmap="viridis",
        square=True,
        cbar=True,
        xticklabels=show_ticks,
        yticklabels=show_ticks,
        ax=ax,
    )
    ax.set_title("Adjacency Matrix Heatmap")
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_flow_graph(
    G: nx.DiGraph,
    out_path: Path,
    population: Optional[np.ndarray] = None,
    show_node_labels: bool = False,
) -> None:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError("Flow graph is empty.")

    pos = nx.spring_layout(G, seed=42)
    weights = np.array([d.get("weight", 0.0) for _, _, d in G.edges(data=True)], dtype=float)
    w_max = weights.max() if weights.size else 1.0
    widths = 0.5 + 4.0 * (weights / w_max) if w_max > 0 else 0.5

    if population is not None and len(population) == G.number_of_nodes():
        pops = population.astype(float)
        p_min = pops.min()
        p_max = pops.max()
        if p_max > p_min:
            sizes = 200 + 1800 * (pops - p_min) / (p_max - p_min)
        else:
            sizes = np.full_like(pops, 400.0)
    else:
        sizes = np.full(G.number_of_nodes(), 400.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4c72b0", alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color="#555555", ax=ax, arrows=True)
    if show_node_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(f"Mobility Flow Graph (edges={G.number_of_edges()})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_timeseries(
    labels: pd.DataFrame,
    nodes: Iterable,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for node in nodes:
        ax.plot(labels.index, labels[node], label=str(node), linewidth=2)

    ax.set_title("Label Time Series (selected nodes)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Label value")
    if len(nodes) <= 10:
        ax.legend(fontsize=8, ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset folder under ./dataset")
    parser.add_argument("--root_path", type=str, default="./dataset/California", help="Root path of the dataset")
    parser.add_argument("--data_path", type=str, default="COVIDinCalifornia.json", help="Data file (json or csv)")
    parser.add_argument("--adj_path", type=str, default=None, help="Adjacency .npy path")
    parser.add_argument("--graph_csv", type=str, default=None, help="Flow graph CSV (src,dst,weight)")
    parser.add_argument("--out_dir", type=str, default="outputs/visualizations", help="Output directory")
    parser.add_argument("--top_k_edges", type=int, default=200, help="Top-K edges to show in flow graph")
    parser.add_argument("--label_nodes", type=str, default=None, help="Comma-separated node names or indices")
    parser.add_argument("--num_label_nodes", type=int, default=5, help="Number of label nodes to auto-select")
    parser.add_argument("--show_node_labels", action="store_true", help="Show node labels on flow graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset:
        root_path = Path("dataset") / args.dataset
        data_path = args.data_path
        if args.data_path is None or args.data_path == "":
            inferred = infer_data_path(root_path)
            if inferred is None:
                raise SystemExit("Unable to infer data file in dataset folder.")
            data_path = inferred
    else:
        root_path = Path(args.root_path)
        data_path = args.data_path

    data_file = root_path / data_path
    if not data_file.exists():
        raise SystemExit(f"Data file not found: {data_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = None
    if data_file.suffix.lower() == ".json":
        labels, payload = load_json_dataset(data_file)
    elif data_file.suffix.lower() == ".csv":
        labels = load_csv_labels(data_file)
    else:
        raise SystemExit("Unsupported data file type. Use .json or .csv.")

    # Try to parse time index
    try:
        labels.index = pd.to_datetime(labels.index)
    except (ValueError, TypeError):
        pass

    # Adjacency matrix
    adj_path = Path(args.adj_path) if args.adj_path else infer_adj_path(root_path)
    adj = None
    if adj_path and adj_path.exists():
        adj = np.load(adj_path)
    elif payload and payload.get("A"):
        adj = np.asarray(payload["A"], dtype=float)
    elif payload and payload.get("G") and payload.get("flow"):
        n = len(payload.get("node", []))
        adj = np.zeros((n, n), dtype=float)
        for (src, dst), w in zip(payload["G"], payload["flow"]):
            adj[int(src), int(dst)] = float(w)

    if adj is None:
        raise SystemExit("Adjacency matrix not found. Provide --adj_path or JSON with A/flow.")

    plot_adj_heatmap(adj, out_dir / "adjacency_heatmap.png")

    # Mobility graph
    if args.graph_csv:
        flow_graph = build_flow_graph_from_csv(Path(args.graph_csv))
        population = None
    elif payload is not None:
        flow_graph, population = build_flow_graph_from_json(payload)
    else:
        flow_graph = None
        population = None

    if flow_graph is not None:
        flow_graph = select_top_edges(flow_graph, args.top_k_edges)
        plot_flow_graph(
            flow_graph,
            out_dir / "mobility_flow_graph.png",
            population=population,
            show_node_labels=args.show_node_labels,
        )

    # Label time series
    selected_nodes = parse_label_nodes(labels, args.label_nodes, args.num_label_nodes)
    plot_label_timeseries(labels, selected_nodes, out_dir / "label_timeseries.png")

    print(f"Saved visualizations to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
