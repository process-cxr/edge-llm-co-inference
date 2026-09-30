#!/usr/bin/env python3
"""
Sweep MATLAB-style rho values and plot the accuracy curve.

This script reuses the model/dataset/pruning utilities defined in `inference.py`,
but sweeps a shared rho across the three device branches. Internally it converts
`rho` to PyTorch unstructured pruning ratio via `pruning_ratio = 1 - rho`.

Examples:
    uv run sweep_pruning.py --rho-start 1 --rho-end 0 --rho-step 0.05 --device cuda
    uv run sweep_pruning.py --start 1 --end 0 --step 0.05 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from . import inference  # noqa: E402
except ImportError:
    import inference  # noqa: E402


@dataclass
class SweepPoint:
    rho: float
    pruning_ratio: float
    acc: float
    loss: float
    sparsity_u1: float
    sparsity_u2: float
    sparsity_u3: float


def _resolve_from_script_dir(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((THIS_DIR / path).resolve())


def _sweep_values(start: float, end: float, step: float) -> List[float]:
    if step == 0:
        raise ValueError("step must be non-zero")

    direction = 1.0 if end >= start else -1.0
    step = abs(step) * direction
    span = end - start
    n = int(round(span / step))

    if n < 0 or abs(start + n * step - end) > 1e-8:
        raise ValueError(
            "start/end must be reachable by step (e.g., 1, 0.95, ..., 0 or 0, 0.05, ..., 1)"
        )

    vals = [round(start + i * step, 10) for i in range(n + 1)]
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"rho must be in [0,1], got {v}")
    return vals


def build_test_loader(
    data_dir: str,
    cache_dir: str,
    image_size: int,
    user_ranges: Tuple[Tuple[float, float], ...],
    imagenet_mean: Tuple[float, float, float],
    imagenet_std: Tuple[float, float, float],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    n_samples: int,
    random_subset: bool,
) -> Tuple[torch.utils.data.DataLoader, int]:
    ds = inference.MultiUserPartialCIFAR10Cached(
        root=data_dir,
        train=False,
        download=True,
        image_size=image_size,
        user_ranges=user_ranges,
        cache_dir=cache_dir,
        fill_mean=imagenet_mean,
        norm_mean=imagenet_mean,
        norm_std=imagenet_std,
    )

    total_test = len(ds)
    if n_samples <= 0 or n_samples >= total_test:
        indices = list(range(total_test))
    else:
        indices = list(range(total_test))
        if random_subset:
            rng = __import__("random").Random(seed)
            rng.shuffle(indices)
        indices = indices[:n_samples]

    subset = torch.utils.data.Subset(ds, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return loader, len(indices)


def load_checkpoint_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep rho and plot the accuracy curve.")

    p.add_argument(
        "--rho-start",
        "--start",
        dest="rho_start",
        type=float,
        default=1.0,
        help="Sweep start for rho in [0,1]. Old alias: --start",
    )
    p.add_argument(
        "--rho-end",
        "--end",
        dest="rho_end",
        type=float,
        default=0.0,
        help="Sweep end for rho in [0,1]. Old alias: --end",
    )
    p.add_argument(
        "--rho-step",
        "--step",
        dest="rho_step",
        type=float,
        default=0.05,
        help="Sweep step for rho. Old alias: --step",
    )
    p.add_argument("--mode", type=str, default=None, choices=[None, "random", "magnitude"])

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--use-amp", action="store_true", help="Use AMP if device=cuda (default: match inference.cfg).")

    p.add_argument("--full-test", action="store_true", help="Evaluate on full CIFAR10 test set (10k).")
    p.add_argument("--num-test-samples", type=int, default=None, help="Override inference.cfg.NUM_TEST_SAMPLES.")
    p.add_argument("--random-test-subset", action="store_true")

    p.add_argument("--ckpt-path", type=str, default=None, help="Override inference.cfg.CKPT_PATH")
    p.add_argument(
        "--out-dir", type=str,
        default=str(THIS_DIR.parent / "outputs" / "image_classification" / "pruning_sweep"),
    )
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--split-layers", type=int, default=None)

    args = p.parse_args()

    inference.set_seed(inference.cfg.SEED)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[Warn] --device cuda requested but CUDA not available; using cpu.")

    ckpt_path_raw = args.ckpt_path or inference.cfg.CKPT_PATH
    ckpt_path = _resolve_from_script_dir(ckpt_path_raw)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    if args.full_test:
        n_samples = -1
    else:
        n_samples = args.num_test_samples if args.num_test_samples is not None else int(inference.cfg.NUM_TEST_SAMPLES)

    split_layers = args.split_layers if args.split_layers is not None else int(inference.cfg.SPLIT_LAYERS)
    prune_mode = (args.mode or inference.cfg.PRUNE_MODE).lower()
    use_amp = bool(args.use_amp) if args.use_amp else bool(inference.cfg.USE_AMP)

    loader, n_eval = build_test_loader(
        data_dir=_resolve_from_script_dir(inference.cfg.DATA_DIR),
        cache_dir=_resolve_from_script_dir(inference.cfg.CACHE_DIR),
        image_size=inference.cfg.IMAGE_SIZE,
        user_ranges=inference.cfg.USER_RANGES,
        imagenet_mean=inference.cfg.IMAGENET_MEAN,
        imagenet_std=inference.cfg.IMAGENET_STD,
        batch_size=inference.cfg.BATCH_SIZE,
        num_workers=inference.cfg.NUM_WORKERS,
        pin_memory=inference.cfg.PIN_MEMORY,
        seed=inference.cfg.SEED,
        n_samples=n_samples,
        random_subset=(args.random_test_subset or bool(inference.cfg.RANDOM_TEST_SUBSET)),
    )

    model = inference.ThreeDeviceSplitViTMeanFusion(
        model_name=inference.cfg.MODEL_NAME,
        num_classes=inference.cfg.NUM_CLASSES,
        split_layers=split_layers,
    ).to(device)
    model.eval()

    state = load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)} (show up to 10): {missing[:10]}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)} (show up to 10): {unexpected[:10]}")
    print(f"[Info] Loaded checkpoint: {ckpt_path}")

    rho_list = _sweep_values(args.rho_start, args.rho_end, args.rho_step)
    points: List[SweepPoint] = []

    print(
        f"[Info] Eval samples: {n_eval} | prune_mode={prune_mode} | "
        f"split_layers={split_layers} | amp={use_amp} | "
        f"rho sweep=({args.rho_start} -> {args.rho_end}, step={args.rho_step})"
    )
    for rho in rho_list:
        pruning_ratio = inference.rho_to_pruning_ratio(rho)

        model.load_state_dict(state, strict=False)

        inference.prune_unstructured(model.device_u1, ratio=pruning_ratio, mode=prune_mode, seed=inference.cfg.SEED + 1)
        inference.prune_unstructured(model.device_u2, ratio=pruning_ratio, mode=prune_mode, seed=inference.cfg.SEED + 2)
        inference.prune_unstructured(model.device_u3, ratio=pruning_ratio, mode=prune_mode, seed=inference.cfg.SEED + 3)

        s1 = inference.compute_sparsity(model.device_u1)
        s2 = inference.compute_sparsity(model.device_u2)
        s3 = inference.compute_sparsity(model.device_u3)

        loss, acc = inference.evaluate(model, loader, device, use_amp)
        points.append(
            SweepPoint(
                rho=rho,
                pruning_ratio=pruning_ratio,
                acc=acc,
                loss=loss,
                sparsity_u1=s1,
                sparsity_u2=s2,
                sparsity_u3=s3,
            )
        )
        print(
            f"rho={rho:>4.2f} | pruning_ratio={pruning_ratio:>4.2f} | "
            f"acc={acc*100:6.2f}% | loss={loss:.4f} | "
            f"sparsity(u1,u2,u3)=({s1*100:5.1f}%, {s2*100:5.1f}%, {s3*100:5.1f}%)"
        )

    csv_path = out_dir / f"pruning_sweep{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "pruning_ratio", "acc_top1", "loss", "sparsity_u1", "sparsity_u2", "sparsity_u3"])
        for pt in points:
            w.writerow([pt.rho, pt.pruning_ratio, pt.acc, pt.loss, pt.sparsity_u1, pt.sparsity_u2, pt.sparsity_u3])

    xs = [pt.rho for pt in points]
    ys = [pt.acc * 100.0 for pt in points]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel("rho (retained fraction, per-device)")
    plt.ylabel("Top-1 accuracy (%)")
    title = f"Accuracy vs rho{(' ' + args.tag) if args.tag else ''}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / f"pruning_sweep_accuracy{suffix}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {fig_path}")


if __name__ == "__main__":
    main()
