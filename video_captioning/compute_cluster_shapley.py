#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
for path in (THIS_DIR, PACKAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .config import cfg  # noqa: E402
    from .dataset_msrvtt import build_default_raw_dataset, collate_raw_video_batch  # noqa: E402
    from .evaluate_caption import compute_caption_metrics  # noqa: E402
    from .inference_caption import default_adapter_dir, default_projector_ckpt, load_server_artifacts  # noqa: E402
    from .models import QwenCaptionModel, ThreeDeviceTinyViTEncoder, set_seed  # noqa: E402
except ImportError:
    from config import cfg  # noqa: E402
    from dataset_msrvtt import build_default_raw_dataset, collate_raw_video_batch  # noqa: E402
    from evaluate_caption import compute_caption_metrics  # noqa: E402
    from inference_caption import default_adapter_dir, default_projector_ckpt, load_server_artifacts  # noqa: E402
    from models import QwenCaptionModel, ThreeDeviceTinyViTEncoder, set_seed  # noqa: E402
from common.shapley_core import COALITIONS, round_robin_partitions, write_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute exact per-cluster Shapley values for Video2TXT.")
    parser.add_argument("--num-samples", type=int, default=160, help="0 or a negative value uses the full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device-model", default=cfg.DEVICE_MODEL_NAME)
    parser.add_argument(
        "--use-4bit", action=argparse.BooleanOptionalAction, default=None,
        help="Use 4-bit Qwen loading (default: project config on CUDA, disabled on CPU).",
    )
    parser.add_argument("--projector-ckpt", default=str(default_projector_ckpt()))
    parser.add_argument("--adapter-dir", default=str(default_adapter_dir()))
    parser.add_argument("--out-dir", default=str(PACKAGE_DIR / "common" / "shapley_outputs"))
    return parser.parse_args()


@torch.no_grad()
def cache_features(
    dataset, indices: List[int], encoder: ThreeDeviceTinyViTEncoder, device: torch.device
) -> List[Tuple[str, torch.Tensor, List[str]]]:
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda", collate_fn=collate_raw_video_batch,
    )
    cached = []
    encoder.eval()
    for batch in loader:
        features = encoder(
            batch["frames_u1"].to(device), batch["frames_u2"].to(device), batch["frames_u3"].to(device)
        ).cpu()
        cached.append((batch["video_ids"][0], features, batch["references"][0]))
    return cached


@torch.no_grad()
def evaluate_cluster(
    cached: List[Tuple[str, torch.Tensor, List[str]]], server: QwenCaptionModel
) -> Dict[int, float]:
    utilities: Dict[int, float] = {}
    for mask in COALITIONS:
        predictions: List[str] = []
        references: List[List[str]] = []
        for _, features, refs in cached:
            masked = features.clone()
            for player in range(3):
                if not mask & (1 << player):
                    masked[:, player, :].zero_()
            prediction = server.generate_from_view_features(
                masked, prompt_texts=[cfg.USER_PROMPT], max_new_tokens=cfg.MAX_NEW_TOKENS,
                num_beams=cfg.NUM_BEAMS,
            )[0]
            predictions.append(prediction)
            references.append(refs)
        utilities[mask] = float(compute_caption_metrics(predictions, references)["CIDEr"])
    return utilities


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_4bit = bool(cfg.USE_4BIT if args.use_4bit is None else args.use_4bit)
    if device.type == "cpu" and use_4bit:
        print("[Warn] 4-bit loading requires CUDA; using full-precision Qwen on CPU.")
        use_4bit = False
    dataset = build_default_raw_dataset(split=args.split, unique_videos=True, max_samples=0)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.num_samples > 0:
        indices = indices[: min(args.num_samples, len(indices))]
    partitions = round_robin_partitions(indices, num_clusters=8)
    if any(not partition for partition in partitions):
        raise ValueError("at least 8 videos are required to populate all clusters")

    encoder = ThreeDeviceTinyViTEncoder(
        model_name=args.device_model, pretrained=cfg.PRETRAINED_DEVICE
    ).to(device)
    server = QwenCaptionModel(
        server_model_name=cfg.SERVER_MODEL_NAME,
        view_feature_dim=int(encoder.feature_dim),
        tokens_per_view=cfg.DEVICE_TOKENS_PER_VIEW,
        use_4bit=use_4bit,
    )
    load_server_artifacts(server, Path(args.projector_ckpt), Path(args.adapter_dir))

    all_utilities: List[Dict[int, float]] = []
    video_ids: List[List[str]] = []
    for cluster_id, cluster_indices in enumerate(partitions, start=1):
        cached = cache_features(dataset, cluster_indices, encoder, device)
        video_ids.append([item[0] for item in cached])
        values = evaluate_cluster(cached, server)
        all_utilities.append(values)
        print(f"[cluster {cluster_id}] samples={len(cached)} utilities={values}")

    metadata = {
        "task": "video",
        "metric": "CIDEr",
        "seed": args.seed,
        "split": args.split,
        "num_samples": len(indices),
        "num_clusters": 8,
        "device_model": args.device_model,
        "server_model": cfg.SERVER_MODEL_NAME,
        "projector_checkpoint": str(Path(args.projector_ckpt).resolve()),
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "device": str(device),
        "use_4bit": use_4bit,
        "coalition_policy": "missing view features are zero; all three visual token slots remain",
        "cluster_video_ids": video_ids,
    }
    write_results(Path(args.out_dir), "video", all_utilities, [len(p) for p in partitions], metadata)
    print(f"[Save] exact Shapley results -> {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
