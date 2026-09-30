#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
for path in (THIS_DIR, PACKAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from . import inference  # noqa: E402
except ImportError:
    import inference  # noqa: E402
from common.shapley_core import COALITIONS, stratified_partitions, write_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute exact per-cluster Shapley values for split ViT.")
    parser.add_argument("--num-samples", type=int, default=10000, help="0 or a negative value uses the full test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--ckpt-path", default=inference.cfg.CKPT_PATH)
    parser.add_argument("--model-name", default=inference.cfg.MODEL_NAME)
    parser.add_argument("--split-layers", type=int, default=inference.cfg.SPLIT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=inference.cfg.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=inference.cfg.NUM_WORKERS)
    parser.add_argument("--out-dir", default=str(PACKAGE_DIR / "common" / "shapley_outputs"))
    return parser.parse_args()


@torch.no_grad()
def evaluate_cluster(
    model: inference.ThreeDeviceSplitViTMeanFusion,
    loader: DataLoader,
    device: torch.device,
) -> Dict[int, float]:
    correct = {mask: 0 for mask in COALITIONS}
    total = 0
    model.eval()
    for x1, x2, x3, targets in loader:
        inputs = [x1.to(device), x2.to(device), x3.to(device)]
        targets = targets.to(device)
        features = [
            model.device_u1(inputs[0]),
            model.device_u2(inputs[1]),
            model.device_u3(inputs[2]),
        ]
        for mask in COALITIONS:
            fused = torch.zeros_like(features[0])
            for player, feature in enumerate(features):
                if mask & (1 << player):
                    fused.add_(feature)
            logits = model.server(fused / 3.0)
            correct[mask] += int((logits.argmax(dim=1) == targets).sum().item())
        total += targets.numel()
    if total == 0:
        raise ValueError("cluster contains no samples")
    return {mask: correct[mask] / total for mask in COALITIONS}


def main() -> None:
    args = parse_args()
    inference.set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = inference.MultiUserPartialCIFAR10Cached(
        root=str((THIS_DIR / inference.cfg.DATA_DIR).resolve()),
        train=False,
        download=False,
        image_size=inference.cfg.IMAGE_SIZE,
        user_ranges=inference.cfg.USER_RANGES,
        cache_dir=str((THIS_DIR / inference.cfg.CACHE_DIR).resolve()),
        fill_mean=inference.cfg.IMAGENET_MEAN,
        norm_mean=inference.cfg.IMAGENET_MEAN,
        norm_std=inference.cfg.IMAGENET_STD,
    )
    count = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
    indices = list(range(count))
    labels = [int(dataset.base.targets[index]) for index in indices]
    partitions = stratified_partitions(indices, labels, seed=args.seed, num_clusters=8)

    model = inference.ThreeDeviceSplitViTMeanFusion(
        model_name=args.model_name,
        num_classes=inference.cfg.NUM_CLASSES,
        split_layers=args.split_layers,
    ).to(device)
    checkpoint = Path(args.ckpt_path)
    if not checkpoint.is_absolute():
        checkpoint = (THIS_DIR / checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)

    utilities: List[Dict[int, float]] = []
    for cluster_id, cluster_indices in enumerate(partitions, start=1):
        loader = DataLoader(
            Subset(dataset, cluster_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        cluster_values = evaluate_cluster(model, loader, device)
        utilities.append(cluster_values)
        print(f"[cluster {cluster_id}] samples={len(cluster_indices)} utilities={cluster_values}")

    metadata = {
        "task": "vit",
        "metric": "top1_accuracy_fraction",
        "seed": args.seed,
        "num_samples": count,
        "num_clusters": 8,
        "checkpoint": str(checkpoint),
        "model_name": args.model_name,
        "split_layers": args.split_layers,
        "device": str(device),
        "coalition_policy": "missing device hidden states are zero; fusion remains sum/3",
        "cluster_indices": partitions,
    }
    write_results(Path(args.out_dir), "vit", utilities, [len(p) for p in partitions], metadata)
    print(f"[Save] exact Shapley results -> {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
