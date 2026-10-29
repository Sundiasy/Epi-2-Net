import argparse
import gc
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from epi2net_model import Epi2NetModel


class SimpleLogger:
    def info(self, msg):
        print(msg)

    def debug(self, msg):
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Epi-2-Net GPU memory with a dummy batch."
    )
    parser.add_argument("--config_filename", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--node_num", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="`auto` prefers CUDA when available.",
    )
    return parser.parse_args()


def format_bytes(num_bytes):
    mib = num_bytes / (1024 ** 2)
    return f"{mib:.2f} MiB"


def get_param_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
    }


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use --device cuda.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runtime_args(cli_args, seq_len, horizon):
    return SimpleNamespace(
        batch_size=int(cli_args.batch_size),
        node_num=int(cli_args.node_num),
        seq_len=int(seq_len),
        label_len=int(horizon),
        token_len=int(seq_len),
        test_seq_len=int(seq_len),
        test_label_len=int(horizon),
        test_pred_len=int(horizon),
    )


def build_dummy_graph(num_nodes):
    adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
    adj /= adj.max()
    return adj


def build_dummy_inputs(seq_len, horizon, batch_size, num_nodes, device, adj_np):
    x = torch.randn(seq_len, batch_size, num_nodes, device=device)
    y = torch.randn(horizon, batch_size, num_nodes, device=device)
    # Keep adj_x on CPU because ODEFunc.forward() converts slices with `.numpy()`.
    adj_x = torch.from_numpy(adj_np).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    return x, y, adj_x


def clear_cuda(device):
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def measure_inference(model, x, y, adj_x, device):
    if device.type != "cuda":
        with torch.no_grad():
            output, _ = model(x, y, adj_x)
        return {
            "output_shape": tuple(output.shape),
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        output, _ = model(x, y, adj_x)
    torch.cuda.synchronize(device)
    return {
        "output_shape": tuple(output.shape),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def measure_training(model, x, y, adj_x, device):
    model.zero_grad(set_to_none=True)

    if device.type != "cuda":
        output, _ = model(x, y, adj_x)
        loss = output.mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        return {
            "loss": float(loss.detach().cpu().item()),
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    output, _ = model(x, y, adj_x)
    loss = output.mean()
    loss.backward()
    torch.cuda.synchronize(device)
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    model.zero_grad(set_to_none=True)
    clear_cuda(device)
    return stats


def main():
    args = parse_args()
    device = resolve_device(args.device)

    with open(args.config_filename, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    model_cfg = dict(config["model"])
    seq_len = args.seq_len if args.seq_len is not None else int(model_cfg["seq_len"])
    horizon = args.horizon if args.horizon is not None else int(model_cfg["horizon"])

    runtime_args = build_runtime_args(args, seq_len=seq_len, horizon=horizon)
    adj_np = build_dummy_graph(num_nodes=args.node_num)
    model = Epi2NetModel(adj_np, SimpleLogger(), runtime_args, **model_cfg).to(device)

    x, y, adj_x = build_dummy_inputs(
        seq_len=seq_len,
        horizon=horizon,
        batch_size=args.batch_size,
        num_nodes=args.node_num,
        device=device,
        adj_np=adj_np,
    )

    stats_before = get_param_stats(model)

    # Warm up once to materialize lazy ODE/GCN parameters before measuring peaks.
    with torch.no_grad():
        _ = model(x, y, adj_x)

    stats_after = get_param_stats(model)
    inference_stats = measure_inference(model, x, y, adj_x, device)
    training_stats = measure_training(model, x, y, adj_x, device)

    input_bytes = (
        x.numel() * x.element_size()
        + y.numel() * y.element_size()
        + adj_x.numel() * adj_x.element_size()
    )
    adam_state_bytes = stats_after["param_bytes"] * 2

    print("=" * 60)
    print("Epi-2-Net memory profile")
    print("=" * 60)
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"node_num: {args.node_num}")
    print(f"seq_len: {seq_len}")
    print(f"horizon: {horizon}")
    print(f"input_dim: {model_cfg.get('input_dim', 1)}")
    print(f"latent_dim: {model_cfg.get('latent_dim')}")
    print(f"filter_type: {model_cfg.get('filter_type')}")
    print()
    print("parameters before warmup:")
    print(f"  total: {stats_before['total_params']:,}")
    print(f"  memory: {format_bytes(stats_before['param_bytes'])}")
    print("parameters after warmup:")
    print(f"  total: {stats_after['total_params']:,}")
    print(f"  memory: {format_bytes(stats_after['param_bytes'])}")
    print(f"  buffers: {format_bytes(stats_after['buffer_bytes'])}")
    print(f"dummy input tensors: {format_bytes(input_bytes)}")
    print(f"adam states estimate: {format_bytes(adam_state_bytes)}")
    print()
    print(f"output shape: {inference_stats['output_shape']}")
    if device.type == "cuda":
        print("inference peak:")
        print(f"  allocated: {format_bytes(inference_stats['peak_allocated_bytes'])}")
        print(f"  reserved: {format_bytes(inference_stats['peak_reserved_bytes'])}")
        print("training peak (forward + backward, no optimizer step):")
        print(f"  allocated: {format_bytes(training_stats['peak_allocated_bytes'])}")
        print(f"  reserved: {format_bytes(training_stats['peak_reserved_bytes'])}")
    else:
        print("CUDA not available, so no real VRAM peak was measured.")
        print("Only parameter/input memory was printed above.")


if __name__ == "__main__":
    main()
