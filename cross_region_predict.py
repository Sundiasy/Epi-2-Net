import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data_provider.data_factory import data_provider, get_adj
from epi2net_model import Epi2NetModel
from metrics import masked_mae_loss, masked_mape_loss, masked_rmse_loss
from utills import load_dataset


class SimpleLogger:
    def info(self, msg):
        print(msg)

    def debug(self, msg):
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load source-region Epi-2-Net weights and run prediction on a target region."
    )
    parser.add_argument("--config_filename", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Source checkpoint path, e.g. models/exp_0/england-epo0.tar")
    parser.add_argument("--source_country", type=str, default="England", help="Only used in logs/output metadata")
    parser.add_argument("--target_country", type=str, default="Italy")
    parser.add_argument("--target_root_path", type=str, default="./dataset/Italy/")
    parser.add_argument("--target_data_path", type=str, default="italy_labels.csv")
    parser.add_argument("--data", type=str, default="epidataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/cross_region")
    parser.add_argument("--prior_graph_mode", type=str, default="distance", choices=["distance", "gravity", "empirical"])
    parser.add_argument("--mobility_graph_mode", type=str, default="empirical", choices=["empirical", "gravity", "distance"])
    parser.add_argument("--gravity_alpha", type=float, default=1.0)
    parser.add_argument("--gravity_beta", type=float, default=1.0)
    parser.add_argument("--gravity_gamma", type=float, default=2.0)
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--label_len", type=int, default=7)
    parser.add_argument("--token_len", type=int, default=7)
    parser.add_argument("--test_seq_len", type=int, default=7)
    parser.add_argument("--test_label_len", type=int, default=7)
    parser.add_argument("--test_pred_len", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_steps", type=str, default="5,7", help="Comma separated horizons for metric reporting.")
    parser.add_argument("--gamma_resize", type=str, default="interpolate", choices=["skip", "mean", "interpolate"])
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--use_cpu_only", action="store_true")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    return parser.parse_args()


def build_model_args(cli_args, node_num):
    runtime_args = argparse.Namespace(**vars(cli_args))
    runtime_args.root_path = cli_args.target_root_path
    runtime_args.data_path = cli_args.target_data_path
    runtime_args.country = cli_args.target_country
    runtime_args.node_num = int(node_num)
    return runtime_args


def tensor_to_device(data, device):
    return data.to(device) if isinstance(data, torch.Tensor) else data


def prepare_batch(batch_x, batch_y, scaler, device):
    x, y = load_dataset(batch_x, batch_y, scaler)
    x = x.permute(1, 0, 2).to(device)
    y = y.permute(1, 0, 2).to(device)
    return x, y


def warmup_dynamic_parameters(model, dataloader, scaler, device):
    try:
        batch_x, _, adj_x, _, batch_y = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("Target dataloader is empty; cannot warm up dynamic graph parameters.") from exc

    x, y = prepare_batch(batch_x, batch_y, scaler, device)
    with torch.no_grad():
        model(x, y, adj_x)


def adapt_gamma_tensor(src_tensor, dst_tensor, mode):
    if mode == "skip":
        return None
    if src_tensor.ndim != 2 or dst_tensor.ndim != 2 or src_tensor.shape[0] != dst_tensor.shape[0]:
        return None
    if mode == "mean":
        return src_tensor.mean(dim=1, keepdim=True).expand_as(dst_tensor)

    resized = F.interpolate(
        src_tensor.unsqueeze(1),
        size=dst_tensor.shape[1],
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return resized


def build_compatible_state_dict(model, checkpoint_state, gamma_resize):
    model_state = model.state_dict()
    compatible_state = {}
    loaded = []
    adapted = []
    skipped = []

    for key, src_tensor in checkpoint_state.items():
        if key not in model_state:
            skipped.append({"key": key, "reason": "missing_in_target"})
            continue

        dst_tensor = model_state[key]
        if src_tensor.shape == dst_tensor.shape:
            compatible_state[key] = src_tensor
            loaded.append(key)
            continue

        if key.endswith("gamma"):
            gamma_tensor = adapt_gamma_tensor(src_tensor, dst_tensor, gamma_resize)
            if gamma_tensor is not None and gamma_tensor.shape == dst_tensor.shape:
                compatible_state[key] = gamma_tensor.to(dtype=dst_tensor.dtype)
                adapted.append(
                    {
                        "key": key,
                        "source_shape": list(src_tensor.shape),
                        "target_shape": list(dst_tensor.shape),
                        "strategy": gamma_resize,
                    }
                )
                continue

        skipped.append(
            {
                "key": key,
                "reason": "shape_mismatch",
                "source_shape": list(src_tensor.shape),
                "target_shape": list(dst_tensor.shape),
            }
        )

    return compatible_state, loaded, adapted, skipped


def inverse_metrics(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return {
        "mae": float(masked_mae_loss(y_pred_inv, y_true_inv).item()),
        "mape": float(masked_mape_loss(y_pred_inv.clone(), y_true_inv.clone()).item()),
        "rmse": float(masked_rmse_loss(y_pred_inv, y_true_inv).item()),
    }, y_true_inv, y_pred_inv


def evaluate(model, dataloader, scaler, device, report_steps):
    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for batch_x, _, adj_x, _, batch_y in dataloader:
            x, y = prepare_batch(batch_x, batch_y, scaler, device)
            output, _ = model(x, y, adj_x)
            preds.append(output.cpu())
            truths.append(y.cpu())

    if not preds:
        raise RuntimeError("No prediction batches were produced for target region.")

    y_pred = torch.cat(preds, dim=1)
    y_true = torch.cat(truths, dim=1)
    full_metrics, y_true_inv, y_pred_inv = inverse_metrics(y_true, y_pred, scaler)

    report = {"full_horizon": full_metrics, "steps": {}}
    horizon = int(y_true.shape[0])
    for step in report_steps:
        effective_step = min(step, horizon)
        step_metrics, _, _ = inverse_metrics(y_true[:effective_step], y_pred[:effective_step], scaler)
        report["steps"][str(step)] = {
            "effective_step": effective_step,
            **step_metrics,
        }

    return report, y_pred_inv.numpy(), y_true_inv.numpy()


def save_outputs(output_dir, metadata, metrics, y_pred, y_true):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "y_preds.npy", y_pred)
    np.save(output_dir / "y_truths.npy", y_true)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cpu" if args.use_cpu_only or not torch.cuda.is_available() else "cuda"
    )

    with open(args.config_filename, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_state = checkpoint["model_state_dict"]

    adj_mx = get_adj(
        args.target_country,
        root_path=args.target_root_path,
        data_path=args.target_data_path,
        graph_mode=args.prior_graph_mode,
        gravity_alpha=args.gravity_alpha,
        gravity_beta=args.gravity_beta,
        gravity_gamma=args.gravity_gamma,
    )
    runtime_args = build_model_args(args, node_num=adj_mx.shape[0])

    _, test_loader, scaler = data_provider(runtime_args, flag="test")
    model = Epi2NetModel(adj_mx, SimpleLogger(), runtime_args, **config["model"]).to(device)

    warmup_dynamic_parameters(model, test_loader, scaler, device)
    compatible_state, loaded_keys, adapted_keys, skipped_keys = build_compatible_state_dict(
        model,
        checkpoint_state,
        gamma_resize=args.gamma_resize,
    )
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)

    report_steps = [
        int(step.strip())
        for step in args.report_steps.split(",")
        if step.strip()
    ]
    metrics, y_pred, y_true = evaluate(model, test_loader, scaler, device, report_steps)

    run_name = f"{args.source_country.lower()}_to_{args.target_country.lower()}"
    output_dir = Path(args.output_dir) / run_name
    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "source_country": args.source_country,
        "target_country": args.target_country,
        "target_root_path": os.path.abspath(args.target_root_path),
        "target_data_path": args.target_data_path,
        "device": str(device),
        "target_num_nodes": int(adj_mx.shape[0]),
        "loaded_key_count": len(loaded_keys),
        "adapted_key_count": len(adapted_keys),
        "skipped_key_count": len(skipped_keys),
        "missing_after_load": list(missing_keys),
        "unexpected_after_load": list(unexpected_keys),
        "adapted_keys": adapted_keys,
        "skipped_keys": skipped_keys,
    }
    save_outputs(output_dir, metadata, metrics, y_pred, y_true)

    print(f"Source: {args.source_country} -> Target: {args.target_country}")
    print(f"Target nodes: {adj_mx.shape[0]}")
    print(f"Loaded exact-match keys: {len(loaded_keys)}")
    print(f"Adapted keys: {len(adapted_keys)}")
    print(f"Skipped keys: {len(skipped_keys)}")
    print(f"Missing after load: {len(missing_keys)}")
    print(f"Unexpected after load: {len(unexpected_keys)}")
    print(f"Full horizon metrics: {metrics['full_horizon']}")
    for step, step_metrics in metrics["steps"].items():
        print(f"Step-{step} metrics: {step_metrics}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
