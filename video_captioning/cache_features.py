from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .config import cfg, ensure_dir, resolve_path
    from .dataset_msrvtt import (
        build_default_raw_dataset,
        build_feature_cache_path,
        collate_raw_video_batch,
    )
    from .models import ThreeDeviceTinyViTEncoder, set_seed
except ImportError:
    from config import cfg, ensure_dir, resolve_path
    from dataset_msrvtt import (
        build_default_raw_dataset,
        build_feature_cache_path,
        collate_raw_video_batch,
    )
    from models import ThreeDeviceTinyViTEncoder, set_seed


def cache_split(split: str, batch_size: int) -> None:
    max_samples = 0
    if split == "train":
        max_samples = cfg.MAX_TRAIN_SAMPLES
    elif split == "val":
        max_samples = cfg.MAX_VAL_SAMPLES
    elif split == "test":
        max_samples = cfg.MAX_TEST_SAMPLES

    dataset = build_default_raw_dataset(split=split, unique_videos=True, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_raw_video_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeDeviceTinyViTEncoder(
        model_name=cfg.DEVICE_MODEL_NAME,
        pretrained=cfg.PRETRAINED_DEVICE,
    ).to(device)
    model.eval()

    feature_root = ensure_dir(cfg.FEATURE_CACHE_DIR)
    print(f"[Info] Caching split={split} to {feature_root}")
    print(f"[Info] device={device} | samples={len(dataset)}")

    with torch.no_grad():
        for batch in loader:
            frames_u1 = batch["frames_u1"].to(device, non_blocking=True)
            frames_u2 = batch["frames_u2"].to(device, non_blocking=True)
            frames_u3 = batch["frames_u3"].to(device, non_blocking=True)
            features = model(frames_u1, frames_u2, frames_u3).cpu()

            for idx, video_id in enumerate(batch["video_ids"]):
                payload = {
                    "video_id": video_id,
                    "view_features": features[idx],
                    "references": batch["references"][idx],
                }
                out_path = build_feature_cache_path(cfg.FEATURE_CACHE_DIR, split, video_id)
                torch.save(payload, out_path)
                print(f"[Cache] {split}/{video_id} -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache TinyViT features for MSR-VTT three-view videos.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    ensure_dir(cfg.CACHE_DIR)
    ensure_dir(cfg.FEATURE_CACHE_DIR)
    set_seed(cfg.SEED)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        cache_split(split=split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
