import argparse
import csv
import difflib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

from cross_region_predict import (
    SimpleLogger,
    build_compatible_state_dict,
    prepare_batch,
    set_seed,
)
from data_provider.data_factory import data_provider, get_adj
from data_provider.graph_utils import (
    build_distance_matrix,
    load_region_dataframe,
    normalize_adjacency_matrix,
)
from epi2net_model import Epi2NetModel


COUNTRY_DATE_RANGES = {
    "italy": ("2020-02-24", "2020-04-24"),
    "england": ("2020-03-13", "2020-05-12"),
    "france": ("2020-03-10", "2020-05-12"),
    "spain": ("2020-03-12", "2020-05-12"),
}

COUNTRY_GRAPH_PREFIX = {
    "italy": "IT",
    "england": "EN",
    "france": "FR",
    "spain": "ES",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Region sensitivity comparison for one city and one 7-day test window. "
            "The script compares baseline prediction against three single-factor "
            "perturbation scenarios: mobility-flow scaling, distance change, and "
            "input-case jump."
        )
    )
    parser.add_argument("--config_filename", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/exp_0/Italy-epo0.tar")
    parser.add_argument("--country", type=str, default="Italy")
    parser.add_argument("--root_path", type=str, default="./dataset/Italy/")
    parser.add_argument("--data_path", type=str, default="italy_labels.csv")
    parser.add_argument("--city", type=str, default="bergamo")
    parser.add_argument("--test_window_index", type=int, default=0)
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Optional prediction start date in YYYY-MM-DD. Overrides --test_window_index.",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/region_perturbation_compare")
    parser.add_argument(
        "--prior_graph_mode",
        type=str,
        default="distance",
        choices=["distance", "gravity", "empirical"],
    )
    parser.add_argument(
        "--mobility_graph_mode",
        type=str,
        default="empirical",
        choices=["empirical", "gravity", "distance"],
    )
    parser.add_argument("--gravity_alpha", type=float, default=1.0)
    parser.add_argument("--gravity_beta", type=float, default=1.0)
    parser.add_argument("--gravity_gamma", type=float, default=2.0)
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--label_len", type=int, default=7)
    parser.add_argument("--token_len", type=int, default=7)
    parser.add_argument("--test_seq_len", type=int, default=7)
    parser.add_argument("--test_label_len", type=int, default=7)
    parser.add_argument("--test_pred_len", type=int, default=7)
    parser.add_argument(
        "--runtime_batch_size",
        type=int,
        default=8,
        help=(
            "Batch size used for inference. The current model registers some dynamic "
            "ODE parameters lazily, so this should match a batch size already seen in the checkpoint."
        ),
    )
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--use_cpu_only", action="store_true")
    parser.add_argument("--gamma_resize", type=str, default="interpolate", choices=["skip", "mean", "interpolate"])
    parser.add_argument("--mobility_scale", type=float, default=20, help="Scale factor for the selected city's dynamic mobility edges.")
    parser.add_argument("--distance_factor", type=float, default=0.05, help="Multiplier applied to the selected city's pairwise distances.")
    parser.add_argument(
        "--cases_jump_ratio",
        type=float,
        default=0.5,
        help="Relative jump applied to the selected city's input cases. 0.5 means +50%%.",
    )
    return parser.parse_args()


def canonical_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_country(country):
    key = str(country).strip().lower()
    country_map = {
        "it": "Italy",
        "italy": "Italy",
        "en": "England",
        "england": "England",
        "fr": "France",
        "france": "France",
        "es": "Spain",
        "spain": "Spain",
    }
    if key not in country_map:
        raise ValueError(
            f"Unsupported country `{country}`. Supported countries: Italy, England, France, Spain."
        )
    return country_map[key]


def load_node_names(root_path, data_path, country=None):
    country_key = normalize_country(country).lower() if country is not None else None
    if country_key in COUNTRY_GRAPH_PREFIX and country_key in COUNTRY_DATE_RANGES:
        start_date, _ = COUNTRY_DATE_RANGES[country_key]
        graph_path = Path(root_path) / "graphs" / f"{COUNTRY_GRAPH_PREFIX[country_key]}_{start_date}.csv"
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                nodes = set()
                for row in reader:
                    if len(row) >= 2:
                        nodes.add(row[0])
                        nodes.add(row[1])
            if nodes:
                return sorted(nodes)

    label_path = Path(root_path) / data_path
    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = [row["name"] for row in reader]
    return sorted(names)


def build_city_lookup(country, root_path, data_path, node_names):
    lookup = {}
    for idx, name in enumerate(node_names):
        lookup[canonical_name(name)] = (idx, name)

    try:
        _, region_df, _ = load_region_dataframe(country=country, root_path=root_path, data_path=data_path)
    except Exception:
        return lookup

    if "name" not in region_df.columns:
        return lookup

    region_df = region_df.copy()
    region_df["_canonical_name"] = region_df["name"].map(canonical_name)
    region_df = region_df.set_index("_canonical_name")

    for idx, name in enumerate(node_names):
        canonical = canonical_name(name)
        if canonical not in region_df.index:
            continue
        row = region_df.loc[canonical]
        if hasattr(row, "ndim") and row.ndim > 1:
            row = row.iloc[0]

        aliases = {name}
        for column in ("name", "name_", "region", "region_name"):
            if column in row.index:
                value = row[column]
                if isinstance(value, str) and value.strip():
                    aliases.add(value)

        for alias in aliases:
            lookup[canonical_name(alias)] = (idx, name)
    return lookup


def resolve_city(city, city_lookup, node_names):
    key = canonical_name(city)
    if key in city_lookup:
        return city_lookup[key]

    suggestions = difflib.get_close_matches(key, list(city_lookup.keys()), n=5, cutoff=0.5)
    raise ValueError(
        f"Unknown city `{city}`. Try one of: {', '.join(suggestions) if suggestions else ', '.join(node_names[:10])}"
    )


def build_dataset_dates(country):
    country_key = normalize_country(country).lower()
    if country_key not in COUNTRY_DATE_RANGES:
        raise ValueError(
            f"No dataset date range configured for `{country}`. Supported countries: {', '.join(sorted(COUNTRY_DATE_RANGES))}"
        )
    start_str, end_str = COUNTRY_DATE_RANGES[country_key]
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    delta_days = (end - start).days
    return [str(start + timedelta(days=i)) for i in range(delta_days + 1)]


def build_test_windows(dataset, dates):
    windows = []
    for idx in range(len(dataset)):
        pred_start = dataset.start_idx + idx + dataset.seq_len
        pred_end = pred_start + dataset.seq_len
        pred_dates = dates[pred_start:pred_end]
        windows.append(
            {
                "index": idx,
                "prediction_start": pred_dates[0],
                "prediction_end": pred_dates[-1],
                "prediction_dates": pred_dates,
            }
        )
    return windows


def resolve_window_index(windows, start_date_arg, country):
    if start_date_arg is None:
        return windows

    for window in windows:
        if window["prediction_start"] == start_date_arg:
            return windows, window["index"]

    valid_dates = ", ".join(window["prediction_start"] for window in windows)
    raise ValueError(
        f"`--start_date {start_date_arg}` is not available in the {normalize_country(country)} test split. "
        f"Available start dates: {valid_dates}"
    )


def infer_supported_batch_sizes(checkpoint_state, latent_dim):
    batch_sizes = set()
    for key, value in checkpoint_state.items():
        if "gconv_adv_weight_" not in key:
            continue
        first_dim = int(value.shape[0])
        if first_dim % latent_dim != 0:
            continue
        num_matrices = first_dim // latent_dim
        if num_matrices < 3:
            continue
        maybe_batch = (num_matrices - 1) / 2
        if maybe_batch.is_integer() and maybe_batch >= 1:
            batch_sizes.add(int(maybe_batch))
    return sorted(batch_sizes)


def clamp_positive(value, name, minimum=1e-6):
    if value <= minimum:
        raise ValueError(f"`{name}` must be larger than {minimum}. Got {value}.")
    return value


def build_runtime_args(args, node_num, runtime_batch_size):
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.root_path = args.root_path
    runtime_args.data_path = args.data_path
    runtime_args.country = args.country
    runtime_args.node_num = int(node_num)
    runtime_args.batch_size = int(runtime_batch_size)
    runtime_args.num_workers = 0
    runtime_args.data = "epidataset"
    runtime_args.use_multi_gpu = False
    runtime_args.local_rank = 0
    return runtime_args


def repeat_single_sample(sample_tensor, batch_size):
    if sample_tensor.dim() < 1:
        raise ValueError("Expected a tensor with at least one dimension.")
    repeats = [batch_size] + [1] * sample_tensor.dim()
    return sample_tensor.unsqueeze(0).repeat(*repeats)


def prepare_repeated_sample(dataset, sample_index, batch_size):
    batch_x, _, adj_x, _, batch_y = dataset[sample_index]
    batch_x = repeat_single_sample(batch_x, batch_size)
    adj_x = repeat_single_sample(adj_x, batch_size)
    batch_y = repeat_single_sample(batch_y, batch_size)
    return batch_x, adj_x, batch_y


def load_model(args):
    with open(args.config_filename, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_state = checkpoint["model_state_dict"]
    supported_batch_sizes = infer_supported_batch_sizes(
        checkpoint_state=checkpoint_state,
        latent_dim=int(config["model"]["latent_dim"]),
    )
    runtime_batch_size = int(args.runtime_batch_size)
    if supported_batch_sizes and runtime_batch_size not in supported_batch_sizes:
        runtime_batch_size = max(supported_batch_sizes)
        print(
            f"Requested runtime batch size {args.runtime_batch_size} is not present in checkpoint. "
            f"Using compatible batch size {runtime_batch_size} instead."
        )

    base_adj = get_adj(
        args.country,
        root_path=args.root_path,
        data_path=args.data_path,
        graph_mode=args.prior_graph_mode,
        gravity_alpha=args.gravity_alpha,
        gravity_beta=args.gravity_beta,
        gravity_gamma=args.gravity_gamma,
    )
    runtime_args = build_runtime_args(args, node_num=base_adj.shape[0], runtime_batch_size=runtime_batch_size)
    test_dataset, _, scaler = data_provider(runtime_args, flag="test")

    device = torch.device("cpu" if args.use_cpu_only or not torch.cuda.is_available() else "cuda")
    model = Epi2NetModel(base_adj, SimpleLogger(), runtime_args, **config["model"]).to(device)

    # The current ODE implementation registers some graph-convolution parameters lazily on first forward.
    warmup_x, warmup_adj, warmup_y = prepare_repeated_sample(test_dataset, 0, runtime_batch_size)
    warmup_inputs, warmup_labels = prepare_batch(warmup_x, warmup_y, scaler, device)
    with torch.no_grad():
        model(warmup_inputs, warmup_labels, warmup_adj)

    compatible_state, loaded_keys, adapted_keys, skipped_keys = build_compatible_state_dict(
        model=model,
        checkpoint_state=checkpoint_state,
        gamma_resize=args.gamma_resize,
    )
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    model.eval()

    return {
        "model": model,
        "device": device,
        "runtime_args": runtime_args,
        "dataset": test_dataset,
        "scaler": scaler,
        "base_adj": np.asarray(base_adj, dtype=float),
        "checkpoint": checkpoint,
        "loaded_keys": loaded_keys,
        "adapted_keys": adapted_keys,
        "skipped_keys": skipped_keys,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "supported_batch_sizes": supported_batch_sizes,
        "runtime_batch_size": runtime_batch_size,
    }


def aggregate_city_prediction(y_pred_inv, y_true_inv, city_idx):
    pred_city = y_pred_inv[:, :, city_idx].mean(axis=1)
    true_city = y_true_inv[:, :, city_idx].mean(axis=1)
    return pred_city.astype(float), true_city.astype(float)


def run_prediction(model, scaler, device, batch_x, batch_y, adj_x, city_idx, seed):
    set_seed(seed)
    x, y = prepare_batch(batch_x, batch_y, scaler, device)
    with torch.no_grad():
        output, _ = model(x, y, adj_x)
    y_pred_inv = scaler.inverse_transform(output.cpu()).numpy()
    y_true_inv = scaler.inverse_transform(y.cpu()).numpy()
    return aggregate_city_prediction(y_pred_inv, y_true_inv, city_idx)


def apply_mobility_perturbation(adj_x, city_idx, mobility_scale):
    mobility_scale = clamp_positive(mobility_scale, "mobility_scale")
    perturbed = adj_x.clone().float()
    perturbed[:, :, city_idx, :] *= mobility_scale
    perturbed[:, :, :, city_idx] *= mobility_scale
    perturbed[:, :, city_idx, city_idx] = adj_x[:, :, city_idx, city_idx]
    return perturbed


def build_distance_perturbed_adj(args, city_name, city_idx, distance_factor):
    distance_factor = clamp_positive(distance_factor, "distance_factor")
    _, region_df, _ = load_region_dataframe(country=args.country, root_path=args.root_path, data_path=args.data_path)
    region_df = region_df.copy()
    if "name" not in region_df.columns:
        raise ValueError("Region dataframe must contain a `name` column.")

    region_df = region_df.sort_values("name").reset_index(drop=True)
    if canonical_name(region_df.loc[city_idx, "name"]) != canonical_name(city_name):
        region_df = region_df.set_index(region_df["name"].map(canonical_name))
        if canonical_name(city_name) not in region_df.index:
            raise ValueError(f"City `{city_name}` is missing from {normalize_country(args.country)} region metadata.")
        ordered_names = [canonical_name(name) for name in load_node_names(args.root_path, args.data_path, args.country)]
        region_df = region_df.loc[ordered_names].reset_index(drop=True)

    distance_matrix = build_distance_matrix(region_df)
    distance_matrix[city_idx, :] *= distance_factor
    distance_matrix[:, city_idx] *= distance_factor
    np.fill_diagonal(distance_matrix, 0.0)

    weight_matrix = 1.0 / (1.0 + distance_matrix + 1e-6)
    perturbed_adj = normalize_adjacency_matrix(weight_matrix, add_self_loops=True, scale_by_max=True)
    return perturbed_adj.astype(float)


def apply_cases_jump(batch_x, city_idx, cases_jump_ratio):
    factor = 1.0 + cases_jump_ratio
    factor = clamp_positive(factor, "1 + cases_jump_ratio")
    perturbed = batch_x.clone().float()
    perturbed[:, :, city_idx, :] *= factor
    perturbed = torch.clamp(perturbed, min=0.0)
    return perturbed


def summarize_scenario(pred_values, baseline_mean):
    mean_value = float(np.mean(pred_values))
    delta = mean_value - baseline_mean
    pct_delta = 0.0 if abs(baseline_mean) < 1e-12 else 100.0 * delta / baseline_mean
    return {
        "daily_pred_cases": [float(x) for x in pred_values],
        "mean_pred_cases_7d": mean_value,
        "delta_vs_baseline": float(delta),
        "pct_delta_vs_baseline": float(pct_delta),
    }


def save_csv(output_path, prediction_dates, truth_daily, baseline_daily, mobility_daily, distance_daily, cases_daily):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "truth_cases",
                "baseline_pred_cases",
                "mobility_pred_cases",
                "mobility_delta_vs_baseline",
                "distance_pred_cases",
                "distance_delta_vs_baseline",
                "cases_jump_pred_cases",
                "cases_jump_delta_vs_baseline",
            ]
        )
        for idx, current_date in enumerate(prediction_dates):
            writer.writerow(
                [
                    current_date,
                    float(truth_daily[idx]),
                    float(baseline_daily[idx]),
                    float(mobility_daily[idx]),
                    float(mobility_daily[idx] - baseline_daily[idx]),
                    float(distance_daily[idx]),
                    float(distance_daily[idx] - baseline_daily[idx]),
                    float(cases_daily[idx]),
                    float(cases_daily[idx] - baseline_daily[idx]),
                ]
            )


def main():
    args = parse_args()
    args.country = normalize_country(args.country)
    set_seed(args.seed)
    if args.use_cpu_only and torch.cuda.is_available():
        raise RuntimeError(
            "`--use_cpu_only` is not supported in the current repository when CUDA is available, "
            "because `epi2net_model.py` and `ode_func.py` bind internal tensors to CUDA at import time. "
            "Run without `--use_cpu_only`, or refactor those modules to make device selection fully runtime-driven."
        )

    artifact = load_model(args)
    model = artifact["model"]
    device = artifact["device"]
    dataset = artifact["dataset"]
    scaler = artifact["scaler"]
    runtime_batch_size = artifact["runtime_batch_size"]

    node_names = load_node_names(args.root_path, args.data_path, args.country)
    city_lookup = build_city_lookup(args.country, args.root_path, args.data_path, node_names)
    city_idx, city_name = resolve_city(args.city, city_lookup, node_names)

    windows = build_test_windows(dataset, build_dataset_dates(args.country))
    if not windows:
        raise RuntimeError(f"No test windows are available for {args.country}.")

    if args.start_date is not None:
        _, window_idx = resolve_window_index(windows, args.start_date, args.country)
    else:
        if args.test_window_index < 0 or args.test_window_index >= len(windows):
            valid_indices = ", ".join(str(window["index"]) for window in windows)
            raise ValueError(
                f"`--test_window_index` must be within the {args.country} test split. Available indices: {valid_indices}"
            )
        window_idx = int(args.test_window_index)

    window_info = windows[window_idx]
    batch_x, adj_x, batch_y = prepare_repeated_sample(dataset, window_idx, runtime_batch_size)

    baseline_daily, truth_daily = run_prediction(
        model=model,
        scaler=scaler,
        device=device,
        batch_x=batch_x,
        batch_y=batch_y,
        adj_x=adj_x,
        city_idx=city_idx,
        seed=args.seed,
    )
    baseline_mean = float(np.mean(baseline_daily))

    mobility_adj_x = apply_mobility_perturbation(adj_x=adj_x, city_idx=city_idx, mobility_scale=args.mobility_scale)
    mobility_daily, _ = run_prediction(
        model=model,
        scaler=scaler,
        device=device,
        batch_x=batch_x,
        batch_y=batch_y,
        adj_x=mobility_adj_x,
        city_idx=city_idx,
        seed=args.seed,
    )

    original_static_adj = np.asarray(model.odefunc.adj_mx, dtype=float).copy()
    distance_daily = None
    try:
        perturbed_static_adj = build_distance_perturbed_adj(
            args=args,
            city_name=city_name,
            city_idx=city_idx,
            distance_factor=args.distance_factor,
        )
        model.odefunc.adj_mx = perturbed_static_adj
        model.adj_mx = perturbed_static_adj
        distance_daily, _ = run_prediction(
            model=model,
            scaler=scaler,
            device=device,
            batch_x=batch_x,
            batch_y=batch_y,
            adj_x=adj_x,
            city_idx=city_idx,
            seed=args.seed,
        )
    finally:
        model.odefunc.adj_mx = original_static_adj
        model.adj_mx = original_static_adj

    cases_batch_x = apply_cases_jump(batch_x=batch_x, city_idx=city_idx, cases_jump_ratio=args.cases_jump_ratio)
    cases_daily, _ = run_prediction(
        model=model,
        scaler=scaler,
        device=device,
        batch_x=cases_batch_x,
        batch_y=batch_y,
        adj_x=adj_x,
        city_idx=city_idx,
        seed=args.seed,
    )

    baseline_summary = {
        "daily_pred_cases": [float(x) for x in baseline_daily],
        "mean_pred_cases_7d": baseline_mean,
    }
    truth_summary = {
        "daily_true_cases": [float(x) for x in truth_daily],
        "mean_true_cases_7d": float(np.mean(truth_daily)),
    }
    mobility_summary = summarize_scenario(mobility_daily, baseline_mean)
    distance_summary = summarize_scenario(distance_daily, baseline_mean)
    cases_summary = summarize_scenario(cases_daily, baseline_mean)

    city_dir = Path(args.output_dir) / args.country.lower() / city_name
    run_dir = city_dir / f"window_{window_idx}_{window_info['prediction_start']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "country": args.country,
            "city": city_name,
            "city_index": city_idx,
            "checkpoint": os.path.abspath(args.checkpoint),
            "runtime_batch_size": runtime_batch_size,
            "supported_batch_sizes_in_checkpoint": artifact["supported_batch_sizes"],
            "prediction_window_index": window_idx,
            "prediction_start": window_info["prediction_start"],
            "prediction_end": window_info["prediction_end"],
            "prediction_dates": window_info["prediction_dates"],
            "prior_graph_mode": args.prior_graph_mode,
            "mobility_graph_mode": args.mobility_graph_mode,
            "seed": args.seed,
            "loaded_key_count": len(artifact["loaded_keys"]),
            "adapted_key_count": len(artifact["adapted_keys"]),
            "skipped_key_count": len(artifact["skipped_keys"]),
            "missing_after_load": artifact["missing_keys"],
            "unexpected_after_load": artifact["unexpected_keys"],
        },
        "assumptions": {
            "mobility_perturbation": {
                "description": "Scale the selected city's incoming and outgoing dynamic mobility edges.",
                "mobility_scale": float(args.mobility_scale),
            },
            "distance_perturbation": {
                "description": "Multiply the selected city's pairwise geographic distances and rebuild the static distance prior graph.",
                "distance_factor": float(args.distance_factor),
            },
            "cases_jump_perturbation": {
                "description": "Scale the selected city's encoder-side input cases.",
                "cases_jump_ratio": float(args.cases_jump_ratio),
            },
        },
        "truth": truth_summary,
        "baseline": baseline_summary,
        "perturbations": {
            "mobility_flow": mobility_summary,
            "distance_change": distance_summary,
            "input_cases_jump": cases_summary,
        },
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    save_csv(
        output_path=run_dir / "daily_predictions.csv",
        prediction_dates=window_info["prediction_dates"],
        truth_daily=truth_daily,
        baseline_daily=baseline_daily,
        mobility_daily=mobility_daily,
        distance_daily=distance_daily,
        cases_daily=cases_daily,
    )

    print(f"City: {city_name} (index={city_idx})")
    print(
        "Prediction window: "
        f"{window_info['prediction_start']} -> {window_info['prediction_end']} "
        f"(test_window_index={window_idx})"
    )
    print(f"Truth 7-day mean cases: {truth_summary['mean_true_cases_7d']:.4f}")
    print(f"Baseline 7-day mean cases: {baseline_summary['mean_pred_cases_7d']:.4f}")
    print(
        "Mobility perturbation 7-day mean cases: "
        f"{mobility_summary['mean_pred_cases_7d']:.4f} "
        f"(delta {mobility_summary['delta_vs_baseline']:+.4f}, "
        f"{mobility_summary['pct_delta_vs_baseline']:+.2f}%)"
    )
    print(
        "Distance perturbation 7-day mean cases: "
        f"{distance_summary['mean_pred_cases_7d']:.4f} "
        f"(delta {distance_summary['delta_vs_baseline']:+.4f}, "
        f"{distance_summary['pct_delta_vs_baseline']:+.2f}%)"
    )
    print(
        "Input-cases jump 7-day mean cases: "
        f"{cases_summary['mean_pred_cases_7d']:.4f} "
        f"(delta {cases_summary['delta_vs_baseline']:+.4f}, "
        f"{cases_summary['pct_delta_vs_baseline']:+.2f}%)"
    )
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
