import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from data_provider.data_factory import get_adj
from epi2net_supervisor import Epi2NetSupervisor
from utills import load_dataset


DATASET_SPECS = {
    "italy": {
        "country": "Italy",
        "root_path": "./dataset/Italy/",
        "data_path": "italy_labels.csv",
        "num_nodes": 105,
    },
    "england": {
        "country": "England",
        "root_path": "./dataset/England/",
        "data_path": "england_labels.csv",
        "num_nodes": 129,
    },
    "france": {
        "country": "France",
        "root_path": "./dataset/France/",
        "data_path": "france_labels.csv",
        "num_nodes": 81,
    },
    "spain": {
        "country": "Spain",
        "root_path": "./dataset/Spain/",
        "data_path": "spain_labels.csv",
        "num_nodes": 34,
    },
    "california": {
        "country": "California",
        "root_path": "./dataset/California/",
        "data_path": "COVIDinCalifornia.json",
        "num_nodes": 58,
    },
    "massachusetts": {
        "country": "Massachusetts",
        "root_path": "./dataset/Massachusetts/",
        "data_path": "COVIDinMassachusetts.json",
        "num_nodes": 14,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def canonicalize_dataset_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().lower().replace("\\", "/").rstrip("/")
    token = normalized.split("/")[-1]
    if token in DATASET_SPECS:
        return token
    for key, spec in DATASET_SPECS.items():
        if token == spec["country"].lower():
            return key
    aliases = {
        "covidincalifornia.json": "california",
        "covidinmassachusetts.json": "massachusetts",
        "italy_labels.csv": "italy",
        "england_labels.csv": "england",
        "france_labels.csv": "france",
        "spain_labels.csv": "spain",
    }
    return aliases.get(token)


def infer_data_path(root_path: Path) -> str | None:
    for pattern in ("*.json", "*_labels.csv"):
        matches = sorted(root_path.glob(pattern))
        if matches:
            return matches[0].name
    return None


def inspect_checkpoint(checkpoint_path: str) -> tuple[dict, int | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"`model_state_dict` not found in checkpoint: {checkpoint_path}")
    state_dict = checkpoint["model_state_dict"]
    gamma = state_dict.get("odefunc.gamma")
    gamma_nodes = int(gamma.shape[-1]) if gamma is not None else None
    return checkpoint, gamma_nodes


def infer_dataset_from_args_or_checkpoint(
    dataset: str | None,
    root_path: str | None,
    data_path: str | None,
    country: str | None,
    checkpoint_nodes: int | None,
) -> str | None:
    for candidate in (dataset, root_path, data_path, None if country == "auto" else country):
        resolved = canonicalize_dataset_name(candidate)
        if resolved:
            return resolved

    if checkpoint_nodes is not None:
        matches = [name for name, spec in DATASET_SPECS.items() if spec["num_nodes"] == checkpoint_nodes]
        if len(matches) == 1:
            return matches[0]
    return None


def resolve_runtime_paths(args: argparse.Namespace, checkpoint_nodes: int | None) -> tuple[str | None, str, str, str]:
    dataset_key = infer_dataset_from_args_or_checkpoint(
        args.dataset, args.root_path, args.data_path, args.country, checkpoint_nodes
    )
    spec = DATASET_SPECS.get(dataset_key) if dataset_key else None

    root_path = args.root_path if args.root_path else (spec["root_path"] if spec else None)
    if root_path is None:
        raise ValueError("Unable to determine root_path. Pass --dataset or --root_path explicitly.")
    root_path = str(root_path)

    data_path = args.data_path if args.data_path else (spec["data_path"] if spec else None)
    if not data_path:
        inferred = infer_data_path(Path(root_path))
        if inferred is None:
            raise ValueError("Unable to determine data_path. Pass --data_path explicitly.")
        data_path = inferred

    if args.country and args.country.lower() != "auto":
        country = args.country
    elif spec:
        country = spec["country"]
    else:
        inferred_country = canonicalize_dataset_name(root_path) or canonicalize_dataset_name(data_path)
        if inferred_country is None:
            raise ValueError("Unable to determine dataset country for adjacency. Pass --country explicitly.")
        country = DATASET_SPECS[inferred_country]["country"]

    return dataset_key, root_path, data_path, country


def load_node_names(root_path: str, data_path: str, num_nodes: int) -> list[str]:
    full_path = Path(root_path) / data_path
    if data_path.lower().endswith(".json") and full_path.exists():
        with full_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        nodes = payload.get("node", [])
        if len(nodes) == num_nodes:
            return [str(node) for node in nodes]
    if data_path.lower().endswith(".csv") and full_path.exists():
        labels = pd.read_csv(full_path)
        if "name" in labels.columns:
            nodes = labels["name"].astype(str).tolist()
            if len(nodes) == num_nodes:
                return nodes
    return [f"node_{idx}" for idx in range(num_nodes)]


def load_model_from_checkpoint(
    supervisor: Epi2NetSupervisor, checkpoint: dict, device: torch.device
) -> None:
    supervisor.pgdenet_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    supervisor.pgdenet_model.to(device)
    supervisor.pgdenet_model.eval()


def collect_test_predictions(supervisor: Epi2NetSupervisor) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for _, (batch_x, _, adj_x, _, batch_y) in enumerate(supervisor.test_dataloader):
            x, y = load_dataset(batch_x, batch_y, supervisor.scaler)
            x, y = supervisor._prepare_data(x, y)
            output, _ = supervisor.pgdenet_model(x, y, adj_x)
            output = supervisor.scaler.inverse_transform(output).detach().cpu()
            preds.append(output)

    if not preds:
        raise RuntimeError("No predictions were generated from the test dataloader.")

    preds_cat = torch.cat(preds, dim=1)
    return preds_cat.numpy()


def extract_gamma(supervisor: Epi2NetSupervisor) -> np.ndarray:
    if not hasattr(supervisor.pgdenet_model.odefunc, "gamma"):
        raise AttributeError("Current model does not define `gamma`. Check `filter_type`.")
    gamma = supervisor.pgdenet_model.odefunc.gamma.detach().cpu().numpy().reshape(-1)
    return gamma


def compute_node_static_statistics(gamma: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, dict]:
    # predictions: [horizon, total_samples, num_nodes]
    if predictions.ndim != 3:
        raise ValueError(f"Expected predictions with 3 dims, got shape {predictions.shape}.")

    node_mean_pred = predictions.mean(axis=(0, 1))
    if gamma.shape[0] != node_mean_pred.shape[0]:
        raise ValueError(
            f"Gamma node count {gamma.shape[0]} does not match predictions node count {node_mean_pred.shape[0]}."
        )

    gamma_std = float(np.std(gamma))
    pred_std = float(np.std(node_mean_pred))

    if gamma_std == 0.0 or pred_std == 0.0:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
        note = "Correlation is undefined because gamma or mean_pred_cases is constant across nodes."
    else:
        pearson_r, pearson_p = stats.pearsonr(gamma, node_mean_pred)
        spearman_r, spearman_p = stats.spearmanr(gamma, node_mean_pred)
        note = ""

    summary = {
        "num_nodes": int(gamma.shape[0]),
        "num_prediction_steps": int(predictions.shape[0]),
        "num_test_windows": int(predictions.shape[1]),
        "gamma_std": gamma_std,
        "mean_pred_cases_std": pred_std,
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p_value": float(spearman_p),
        "note": note,
    }
    return node_mean_pred, summary


def save_outputs(
    output_dir: str,
    node_names: list[str],
    gamma: np.ndarray,
    node_mean_pred: np.ndarray,
    predictions: np.ndarray,
    summary: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    table = pd.DataFrame(
        {
            "node_idx": np.arange(len(node_names)),
            "node_name": node_names,
            "gamma": gamma,
            "mean_pred_cases": node_mean_pred,
        }
    )
    table.to_csv(os.path.join(output_dir, "gamma_node_static.csv"), index=False)
    np.save(os.path.join(output_dir, "gamma.npy"), gamma)
    np.save(os.path.join(output_dir, "pred_cases_test.npy"), predictions)

    with open(os.path.join(output_dir, "gamma_correlation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "gamma_correlation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Node-level static correlation between gamma and predicted cases\n")
        f.write(f"num_nodes: {summary['num_nodes']}\n")
        f.write(f"num_prediction_steps: {summary['num_prediction_steps']}\n")
        f.write(f"num_test_windows: {summary['num_test_windows']}\n")
        f.write(f"dataset: {summary['dataset']}\n")
        f.write(f"country: {summary['country']}\n")
        f.write(f"root_path: {summary['root_path']}\n")
        f.write(f"data_path: {summary['data_path']}\n")
        f.write(f"pearson_r: {summary['pearson_r']:.6f}\n")
        f.write(f"pearson_p_value: {summary['pearson_p_value']:.6e}\n")
        f.write(f"spearman_r: {summary['spearman_r']:.6f}\n")
        f.write(f"spearman_p_value: {summary['spearman_p_value']:.6e}\n")
        if summary["note"]:
            f.write(f"note: {summary['note']}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="config.yaml", type=str)
    parser.add_argument("--checkpoint", default="models/exp_0/epo0.tar", type=str)
    parser.add_argument("--output_dir", default="outputs/gamma_correlation", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--country", default="auto", type=str)
    parser.add_argument("--seed", default=2035, type=int)

    parser.add_argument("--use_cpu_only", default=False, type=bool)
    parser.add_argument("--exp_repeat", type=int, default=1)
    parser.add_argument("--data", type=str, default="epidataset")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--node_num", type=int, default=-1)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--label_len", type=int, default=7)
    parser.add_argument("--token_len", type=int, default=7)
    parser.add_argument("--test_seq_len", type=int, default=7)
    parser.add_argument("--test_label_len", type=int, default=7)
    parser.add_argument("--test_pred_len", type=int, default=7)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    return parser


def main() -> None:
    args, _ = build_arg_parser().parse_known_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.use_cpu_only or not torch.cuda.is_available() else "cuda")
    checkpoint, checkpoint_nodes = inspect_checkpoint(args.checkpoint)
    dataset_key, root_path, data_path, country = resolve_runtime_paths(args, checkpoint_nodes)
    args.root_path = root_path
    args.data_path = data_path

    with open(args.config_filename, "r", encoding="utf-8") as f:
        supervisor_config = yaml.load(f, Loader=yaml.Loader)

    adj_mx = get_adj(country)
    if checkpoint_nodes is not None and checkpoint_nodes != int(adj_mx.shape[0]):
        raise ValueError(
            f"Checkpoint expects {checkpoint_nodes} nodes, but dataset `{country}` provides {adj_mx.shape[0]} nodes. "
            f"Use matching --dataset/--root_path/--data_path for this checkpoint."
        )
    args.node_num = int(adj_mx.shape[0])
    supervisor = Epi2NetSupervisor(adj_mx=adj_mx, args=args, **supervisor_config)
    load_model_from_checkpoint(supervisor, checkpoint, device)

    gamma = extract_gamma(supervisor)
    predictions = collect_test_predictions(supervisor)
    node_names = load_node_names(args.root_path, args.data_path, len(gamma))
    node_mean_pred, summary = compute_node_static_statistics(gamma, predictions)
    summary.update(
        {
            "dataset": dataset_key or "custom",
            "country": country,
            "root_path": args.root_path,
            "data_path": args.data_path,
        }
    )
    save_outputs(args.output_dir, node_names, gamma, node_mean_pred, predictions, summary)

    print("Node-level static correlation between gamma and predicted cases")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {args.output_dir}")
    print(f"dataset: {summary['dataset']} | country: {country}")
    print(f"root_path: {args.root_path}")
    print(f"data_path: {args.data_path}")
    print(f"pearson_r: {summary['pearson_r']:.6f} (p={summary['pearson_p_value']:.6e})")
    print(f"spearman_r: {summary['spearman_r']:.6f} (p={summary['spearman_p_value']:.6e})")


if __name__ == "__main__":
    main()
