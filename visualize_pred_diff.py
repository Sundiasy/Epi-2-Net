"""
Visualize differences between prediction and truth .npy arrays.

Example:
  python visualize_pred_diff.py --dataset Italy
  python visualize_pred_diff.py --pred dataset/Italy/y_preds.npy --truth dataset/Italy/y_truths.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# TODO: Fill in mean/std for de-standardization if needed.
# You can use a scalar, or a numpy array that broadcasts to your data shape.
DATA_MEAN = 1286.229850
DATA_STD = 6674.404178


def maybe_denormalize(arr: np.ndarray) -> np.ndarray:
    if DATA_MEAN is None or DATA_STD is None:
        return arr
    return arr * DATA_STD + DATA_MEAN


def infer_axes(
    arr: np.ndarray,
    time_axis: int | None,
    node_axis: int | None,
    max_time: int,
) -> tuple[int, int | None]:
    if arr.ndim == 1:
        return 0, None
    if arr.ndim == 2:
        return 0, 1

    if time_axis is None:
        small_axes = [i for i, s in enumerate(arr.shape) if s <= max_time]
        if small_axes:
            time_axis = min(small_axes, key=lambda i: arr.shape[i])
        else:
            time_axis = 1

    if node_axis is None:
        remaining = [i for i in range(arr.ndim) if i != time_axis]
        if len(remaining) == 1:
            node_axis = remaining[0]
        else:
            node_axis = min(remaining, key=lambda i: arr.shape[i])

    return time_axis, node_axis


def to_samples_time_nodes(
    arr: np.ndarray,
    time_axis: int | None,
    node_axis: int | None,
    max_time: int,
) -> tuple[np.ndarray, dict]:
    arr = np.asarray(arr)

    if arr.ndim == 1:
        out = arr[None, :, None]
        return out, {"time_axis": 0, "node_axis": None, "sample_axes": []}

    if arr.ndim == 2:
        out = arr[None, :, :]
        return out, {"time_axis": 0, "node_axis": 1, "sample_axes": []}

    time_axis, node_axis = infer_axes(arr, time_axis, node_axis, max_time)
    if time_axis == node_axis:
        raise ValueError("time_axis and node_axis must be different.")

    sample_axes = [i for i in range(arr.ndim) if i not in (time_axis, node_axis)]
    new_order = sample_axes + [time_axis, node_axis]
    arr = np.transpose(arr, axes=new_order)

    if sample_axes:
        sample_size = int(np.prod(arr.shape[: len(sample_axes)]))
    else:
        sample_size = 1
    time_size = arr.shape[len(sample_axes)]
    node_size = arr.shape[len(sample_axes) + 1]

    out = arr.reshape(sample_size, time_size, node_size)
    return out, {"time_axis": time_axis, "node_axis": node_axis, "sample_axes": sample_axes}


def find_pairs(root: Path, pred_name: str, truth_name: str) -> list[tuple[str, Path, Path]]:
    pairs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        pred = path / pred_name
        truth = path / truth_name
        if pred.exists() and truth.exists():
            pairs.append((path.name, pred, truth))
    return pairs


def load_arrays(pred_path: Path, truth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(pred_path)
    truth = np.load(truth_path)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs truth {truth.shape}")
    return pred, truth


def plot_differences(
    pred: np.ndarray,
    truth: np.ndarray,
    out_path: Path,
    time_axis: int | None,
    node_axis: int | None,
    max_time: int,
    max_points: int,
) -> None:
    pred = maybe_denormalize(pred)
    truth = maybe_denormalize(truth)

    pred_stn, info = to_samples_time_nodes(pred, time_axis, node_axis, max_time=max_time)
    truth_stn, _ = to_samples_time_nodes(truth, info["time_axis"], info["node_axis"], max_time=max_time)

    diff = pred_stn - truth_stn
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    mean_pred = np.mean(pred_stn, axis=0)   # (time, nodes)
    mean_truth = np.mean(truth_stn, axis=0)
    mean_abs_diff = np.mean(np.abs(diff), axis=0)

    time_len, node_len = mean_truth.shape
    node_indices = np.linspace(0, node_len - 1, num=min(3, node_len), dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1) Mean trajectories for a few nodes
    ax = axes[0, 0]
    for idx in node_indices:
        ax.plot(mean_truth[:, idx], label=f"truth node {idx}", linewidth=2)
        ax.plot(mean_pred[:, idx], label=f"pred node {idx}", linestyle="--")
    ax.set_title("Mean trajectories (selected nodes)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, ncol=2)

    # 2) Heatmap of mean absolute error over time x nodes
    ax = axes[0, 1]
    im = ax.imshow(mean_abs_diff, aspect="auto", origin="lower")
    ax.set_title("Mean |pred - truth| (time x node)")
    ax.set_xlabel("Node")
    ax.set_ylabel("Time")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3) Error histogram
    ax = axes[1, 0]
    ax.hist(diff.ravel(), bins=50, color="#4c72b0", alpha=0.8)
    ax.set_title("Error distribution (pred - truth)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")

    # 4) Pred vs truth scatter
    ax = axes[1, 1]
    flat_truth = truth_stn.ravel()
    flat_pred = pred_stn.ravel()
    if flat_truth.size > max_points:
        idx = np.random.choice(flat_truth.size, size=max_points, replace=False)
        flat_truth = flat_truth[idx]
        flat_pred = flat_pred[idx]
    ax.scatter(flat_truth, flat_pred, s=5, alpha=0.4)
    min_v = min(flat_truth.min(), flat_pred.min())
    max_v = max(flat_truth.max(), flat_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], color="black", linewidth=1)
    ax.set_title("Pred vs Truth (sampled)")
    ax.set_xlabel("Truth")
    ax.set_ylabel("Pred")

    fig.suptitle(f"MAE={mae:.4f} | RMSE={rmse:.4f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=None, help="Path to prediction .npy file")
    parser.add_argument("--truth", type=str, default=None, help="Path to truth .npy file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset folder name under ./dataset/")
    parser.add_argument("--pred-name", type=str, default="y_preds.npy", help="Filename for prediction")
    parser.add_argument("--truth-name", type=str, default="y_truths.npy", help="Filename for truth")
    parser.add_argument("--out", type=str, default="outputs/vis_diff.png", help="Output image path")
    parser.add_argument("--time-axis", type=int, default=None, help="Axis index for time dimension")
    parser.add_argument("--node-axis", type=int, default=None, help="Axis index for node dimension")
    parser.add_argument("--max-time", type=int, default=32, help="Max size to infer time axis")
    parser.add_argument("--max-points", type=int, default=5000, help="Max points for scatter plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path("dataset")

    if args.pred and args.truth:
        pred_path = Path(args.pred)
        truth_path = Path(args.truth)
    elif args.dataset:
        pred_path = root / args.dataset / args.pred_name
        truth_path = root / args.dataset / args.truth_name
    else:
        pairs = find_pairs(root, args.pred_name, args.truth_name)
        if len(pairs) == 1:
            _, pred_path, truth_path = pairs[0]
        else:
            msg = ["Please specify --dataset or --pred/--truth. Found pairs:"]
            for name, pred, truth in pairs:
                msg.append(f"  {name}: {pred} | {truth}")
            raise SystemExit("\n".join(msg))

    pred, truth = load_arrays(pred_path, truth_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_differences(
        pred,
        truth,
        out_path,
        time_axis=args.time_axis,
        node_axis=args.node_axis,
        max_time=args.max_time,
        max_points=args.max_points,
    )
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
