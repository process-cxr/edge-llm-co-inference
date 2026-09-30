from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from .config import cfg, ensure_dir, resolve_path
    from .dataset_msrvtt import (
        build_default_cached_dataset,
        build_default_raw_dataset,
        collate_cached_feature_batch,
        collate_raw_video_batch,
        load_or_sample_video_frames,
        make_partial_view_video,
    )
    from .models import (
        QwenCaptionModel,
        ThreeDeviceTinyViTEncoder,
        compute_sparsity,
        prune_unstructured,
        rho_to_pruning_ratio,
        set_seed,
    )
except ImportError:
    from config import cfg, ensure_dir, resolve_path
    from dataset_msrvtt import (
        build_default_cached_dataset,
        build_default_raw_dataset,
        collate_cached_feature_batch,
        collate_raw_video_batch,
        load_or_sample_video_frames,
        make_partial_view_video,
    )
    from models import (
        QwenCaptionModel,
        ThreeDeviceTinyViTEncoder,
        compute_sparsity,
        prune_unstructured,
        rho_to_pruning_ratio,
        set_seed,
    )


def parse_csv_floats(arg: str, expected_len: int, name: str) -> List[float]:
    values = [float(item.strip()) for item in arg.split(",") if item.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{name} must provide exactly {expected_len} comma-separated values, got {arg!r}")
    return values


def format_float_list(values: Sequence[float]) -> str:
    return ",".join(f"{value:.4f}" for value in values)


def resolve_sv_weights(
    sv_weights_arg: str,
    scale_mode: str,
    num_views: int,
) -> Tuple[List[float], torch.Tensor]:
    if sv_weights_arg:
        raw_weights = parse_csv_floats(sv_weights_arg, expected_len=num_views, name="sv_weights")
    else:
        raw_weights = [1.0] * num_views

    if any(weight < 0.0 for weight in raw_weights):
        raise ValueError(f"sv_weights must be non-negative, got {raw_weights}")

    raw_tensor = torch.tensor(raw_weights, dtype=torch.float32)
    if raw_tensor.sum().item() <= 0.0:
        raise ValueError("sv_weights cannot all be zero")

    if scale_mode == "none":
        applied = raw_tensor
    elif scale_mode == "sum1":
        applied = raw_tensor / raw_tensor.sum()
    elif scale_mode == "mean1":
        applied = raw_tensor * (num_views / raw_tensor.sum())
    else:
        raise ValueError(f"Unsupported sv scale mode: {scale_mode}")

    return raw_weights, applied


def apply_view_weights(view_features: torch.Tensor, sv_weights: torch.Tensor) -> torch.Tensor:
    if view_features.ndim != 3:
        raise ValueError(f"view_features must have shape [B, V, D], got {tuple(view_features.shape)}")
    if view_features.shape[1] != sv_weights.numel():
        raise ValueError(
            f"sv_weights length {sv_weights.numel()} does not match num_views {view_features.shape[1]}"
        )
    weights = sv_weights.to(device=view_features.device, dtype=view_features.dtype).view(1, -1, 1)
    return view_features * weights


def default_projector_ckpt() -> Path:
    generated = resolve_path(cfg.SAVE_DIR) / "qlora" / cfg.PROJECTOR_CKPT_NAME
    packaged = Path(__file__).resolve().parent / "checkpoints" / "qlora" / cfg.PROJECTOR_CKPT_NAME
    if generated.exists():
        return generated
    return packaged


def default_adapter_dir() -> Path:
    generated = resolve_path(cfg.SAVE_DIR) / "qlora" / cfg.QLORA_DIR_NAME
    packaged = Path(__file__).resolve().parent / "checkpoints" / "qlora" / cfg.QLORA_DIR_NAME
    if generated.exists():
        return generated
    return packaged


def load_server_artifacts(model: QwenCaptionModel, projector_ckpt: Path, adapter_dir: Path) -> None:
    if projector_ckpt.exists():
        model.load_projector_checkpoint(projector_ckpt)
    else:
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_ckpt}")

    if adapter_dir.exists():
        from peft import PeftModel

        model.lm = PeftModel.from_pretrained(model.lm, adapter_dir, is_trainable=False)
        print(f"[Info] Loaded LoRA adapter: {adapter_dir}")
    else:
        print(f"[Warn] Adapter dir not found, using projector-only model: {adapter_dir}")


def load_single_video_views(video_path: Path) -> Dict[str, torch.Tensor]:
    frames_u8 = load_or_sample_video_frames(
        video_path=video_path,
        num_frames=cfg.NUM_FRAMES,
        frame_size=cfg.FRAME_SIZE,
    )
    views = []
    for left, right in cfg.USER_RANGES:
        views.append(
            make_partial_view_video(
                frames_u8=frames_u8,
                left=left,
                right=right,
                fill_mean=cfg.IMAGENET_MEAN,
                norm_mean=cfg.IMAGENET_MEAN,
                norm_std=cfg.IMAGENET_STD,
            )
        )
    return {
        "frames_u1": views[0].unsqueeze(0),
        "frames_u2": views[1].unsqueeze(0),
        "frames_u3": views[2].unsqueeze(0),
    }


def build_device_encoder(rho1: float, rho2: float, rho3: float) -> ThreeDeviceTinyViTEncoder:
    encoder = ThreeDeviceTinyViTEncoder(
        model_name=cfg.DEVICE_MODEL_NAME,
        pretrained=cfg.PRETRAINED_DEVICE,
    )
    encoder.eval()

    prune_unstructured(
        encoder.device_u1,
        ratio=rho_to_pruning_ratio(rho1),
        mode=cfg.PRUNE_MODE,
        seed=cfg.SEED + 1,
    )
    prune_unstructured(
        encoder.device_u2,
        ratio=rho_to_pruning_ratio(rho2),
        mode=cfg.PRUNE_MODE,
        seed=cfg.SEED + 2,
    )
    prune_unstructured(
        encoder.device_u3,
        ratio=rho_to_pruning_ratio(rho3),
        mode=cfg.PRUNE_MODE,
        seed=cfg.SEED + 3,
    )
    return encoder


def run_dataset_inference(
    model: QwenCaptionModel,
    input_mode: str,
    split: str,
    num_samples: int,
    device_encoder: ThreeDeviceTinyViTEncoder | None,
    sv_weights: torch.Tensor,
    rho_values: Sequence[float],
    sv_weights_raw: Sequence[float],
    sv_scale_mode: str,
) -> List[Dict[str, object]]:
    if input_mode == "cached":
        dataset = build_default_cached_dataset(split=split, unique_videos=True, max_samples=num_samples)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_cached_feature_batch,
        )
    else:
        dataset = build_default_raw_dataset(split=split, unique_videos=True, max_samples=num_samples)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_raw_video_batch,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[Dict[str, object]] = []
    if device_encoder is not None:
        device_encoder = device_encoder.to(device)
        device_encoder.eval()

    for batch in loader:
        start = time.perf_counter()
        if input_mode == "cached":
            view_features = batch["view_features"]
        else:
            assert device_encoder is not None
            frames_u1 = batch["frames_u1"].to(device, non_blocking=True)
            frames_u2 = batch["frames_u2"].to(device, non_blocking=True)
            frames_u3 = batch["frames_u3"].to(device, non_blocking=True)
            with torch.no_grad():
                view_features = device_encoder(frames_u1, frames_u2, frames_u3).cpu()

        view_features = apply_view_weights(view_features, sv_weights)

        predictions = model.generate_from_view_features(
            view_features=view_features,
            prompt_texts=[cfg.USER_PROMPT],
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            num_beams=cfg.NUM_BEAMS,
        )
        latency = time.perf_counter() - start
        results.append(
            {
                "video_id": batch["video_ids"][0],
                "prediction": predictions[0],
                "references": batch["references"][0],
                "latency_sec": latency,
                "upload_tokens": 3 * cfg.DEVICE_TOKENS_PER_VIEW,
                "rho": list(rho_values),
                "sv_weights_raw": list(sv_weights_raw),
                "sv_weights_applied": sv_weights.tolist(),
                "sv_scale_mode": sv_scale_mode,
            }
        )
        print(f"[Pred] {batch['video_ids'][0]} | {predictions[0]}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run video caption inference.")
    parser.add_argument("--input-mode", choices=["raw", "cached"], default="raw")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--video-path", type=str, default="")
    parser.add_argument("--projector-ckpt", type=str, default="")
    parser.add_argument("--adapter-dir", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--rho3", type=float, default=1.0)
    parser.add_argument("--sv-weights", type=str, default="")
    parser.add_argument("--sv-scale-mode", choices=["none", "sum1", "mean1"], default="mean1")
    args = parser.parse_args()

    set_seed(cfg.SEED)
    projector_ckpt = Path(args.projector_ckpt) if args.projector_ckpt else default_projector_ckpt()
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else default_adapter_dir()
    rho_values = [args.rho1, args.rho2, args.rho3]
    sv_weights_raw, sv_weights = resolve_sv_weights(
        sv_weights_arg=args.sv_weights,
        scale_mode=args.sv_scale_mode,
        num_views=len(cfg.USER_RANGES),
    )

    feature_dim: int
    if args.input_mode == "cached":
        sample = build_default_cached_dataset(split=args.split, unique_videos=True, max_samples=1)[0]
        feature_dim = int(sample["view_features"].shape[-1])
    else:
        probe_encoder = ThreeDeviceTinyViTEncoder(
            model_name=cfg.DEVICE_MODEL_NAME,
            pretrained=cfg.PRETRAINED_DEVICE,
        )
        feature_dim = int(probe_encoder.feature_dim)
        del probe_encoder

    model = QwenCaptionModel(
        server_model_name=cfg.SERVER_MODEL_NAME,
        view_feature_dim=feature_dim,
        tokens_per_view=cfg.DEVICE_TOKENS_PER_VIEW,
        use_4bit=cfg.USE_4BIT,
    )
    load_server_artifacts(model, projector_ckpt=projector_ckpt, adapter_dir=adapter_dir)
    print(
        "[Info] simulation config | "
        f"rho=({format_float_list(rho_values)}) | "
        f"sv_raw=({format_float_list(sv_weights_raw)}) | "
        f"sv_applied[{args.sv_scale_mode}]=({format_float_list(sv_weights.tolist())})"
    )

    if args.video_path:
        if args.input_mode != "raw":
            raise ValueError("--video-path currently requires --input-mode raw")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_encoder = build_device_encoder(args.rho1, args.rho2, args.rho3).to(device)
        batch = load_single_video_views(Path(args.video_path))
        with torch.no_grad():
            view_features = device_encoder(
                batch["frames_u1"].to(device),
                batch["frames_u2"].to(device),
                batch["frames_u3"].to(device),
            ).cpu()
        view_features = apply_view_weights(view_features, sv_weights)
        predictions = model.generate_from_view_features(view_features, prompt_texts=[cfg.USER_PROMPT])
        print(predictions[0])
        return

    device_encoder = None
    if args.input_mode == "raw":
        device_encoder = build_device_encoder(args.rho1, args.rho2, args.rho3)
        print(
            "[Info] device sparsity: "
            f"u1={compute_sparsity(device_encoder.device_u1)*100:.2f}% | "
            f"u2={compute_sparsity(device_encoder.device_u2)*100:.2f}% | "
            f"u3={compute_sparsity(device_encoder.device_u3)*100:.2f}%"
        )

    results = run_dataset_inference(
        model=model,
        input_mode=args.input_mode,
        split=args.split,
        num_samples=args.num_samples,
        device_encoder=device_encoder,
        sv_weights=sv_weights,
        rho_values=rho_values,
        sv_weights_raw=sv_weights_raw,
        sv_scale_mode=args.sv_scale_mode,
    )
    out_dir = ensure_dir(cfg.PREDICTION_DIR)
    output_file = Path(args.output_file) if args.output_file else out_dir / f"{args.split}_predictions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    print(f"[Save] predictions -> {output_file}")


if __name__ == "__main__":
    main()
