from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import cfg, ensure_dir, resolve_path
    from .dataset_msrvtt import build_default_raw_dataset, collate_raw_video_batch
    from .evaluate_caption import compute_caption_metrics
    from .inference_caption import (apply_view_weights, default_adapter_dir, default_projector_ckpt,
        format_float_list, load_server_artifacts, resolve_sv_weights)
    from .models import (QwenCaptionModel, ThreeDeviceTinyViTEncoder, compute_sparsity,
        prune_unstructured, rho_to_pruning_ratio, set_seed)
except ImportError:
    from config import cfg, ensure_dir, resolve_path
    from dataset_msrvtt import build_default_raw_dataset, collate_raw_video_batch
    from evaluate_caption import compute_caption_metrics
    from inference_caption import (apply_view_weights, default_adapter_dir, default_projector_ckpt,
        format_float_list, load_server_artifacts, resolve_sv_weights)
    from models import (QwenCaptionModel, ThreeDeviceTinyViTEncoder, compute_sparsity,
        prune_unstructured, rho_to_pruning_ratio, set_seed)


def parse_rho_list(arg: str) -> List[float]:
    if not arg:
        return list(cfg.RHO_LIST)
    return [float(value) for value in arg.split(",")]


def parse_rho_triplets(arg: str) -> List[List[float]]:
    triplets: List[List[float]] = []
    if not arg.strip():
        return triplets

    for case_text in arg.split(";"):
        case_text = case_text.strip()
        if not case_text:
            continue
        values = [float(item.strip()) for item in case_text.split(",") if item.strip()]
        if len(values) != 3:
            raise ValueError(
                f"Each rho triplet must provide exactly 3 values like '0.8,0.9,1.0', got {case_text!r}"
            )
        triplets.append(values)
    return triplets


def resolve_rho_cases(rho_list_arg: str, rho_triplets_arg: str) -> List[List[float]]:
    triplets = parse_rho_triplets(rho_triplets_arg)
    if triplets:
        return triplets
    return [[rho, rho, rho] for rho in parse_rho_list(rho_list_arg)]


def is_shared_rho_case(rho_values: List[float]) -> bool:
    return max(rho_values) - min(rho_values) < 1e-9


def plot_curve(xs: List[float], ys: List[float], ylabel: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("rho")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep device pruning ratios for video captioning.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--rho-list", type=str, default="")
    parser.add_argument("--rho-triplets", type=str, default="")
    parser.add_argument("--sv-weights", type=str, default="")
    parser.add_argument("--sv-scale-mode", choices=["none", "sum1", "mean1"], default="mean1")
    parser.add_argument(
        "--out-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "video_captioning" / "pruning_sweep"),
    )
    parser.add_argument("--projector-ckpt", type=str, default="")
    parser.add_argument("--adapter-dir", type=str, default="")
    args = parser.parse_args()

    set_seed(cfg.SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_default_raw_dataset(split=args.split, unique_videos=True, max_samples=args.num_samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_raw_video_batch,
    )

    base_encoder = ThreeDeviceTinyViTEncoder(
        model_name=cfg.DEVICE_MODEL_NAME,
        pretrained=cfg.PRETRAINED_DEVICE,
    )
    base_state = base_encoder.state_dict()
    feature_dim = int(base_encoder.feature_dim)

    projector_ckpt = Path(args.projector_ckpt) if args.projector_ckpt else default_projector_ckpt()
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else default_adapter_dir()
    server_model = QwenCaptionModel(
        server_model_name=cfg.SERVER_MODEL_NAME,
        view_feature_dim=feature_dim,
        tokens_per_view=cfg.DEVICE_TOKENS_PER_VIEW,
        use_4bit=cfg.USE_4BIT,
    )
    load_server_artifacts(server_model, projector_ckpt=projector_ckpt, adapter_dir=adapter_dir)

    rho_cases = resolve_rho_cases(args.rho_list, args.rho_triplets)
    sv_weights_raw, sv_weights = resolve_sv_weights(
        sv_weights_arg=args.sv_weights,
        scale_mode=args.sv_scale_mode,
        num_views=3,
    )
    rows: List[Dict[str, float]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_rho_only = all(is_shared_rho_case(rho_values) for rho_values in rho_cases)

    print(
        "[Info] sweep config | "
        f"num_cases={len(rho_cases)} | "
        f"sv_raw=({format_float_list(sv_weights_raw)}) | "
        f"sv_applied[{args.sv_scale_mode}]=({format_float_list(sv_weights.tolist())})"
    )

    for case_idx, rho_values in enumerate(rho_cases, start=1):
        encoder = ThreeDeviceTinyViTEncoder(
            model_name=cfg.DEVICE_MODEL_NAME,
            pretrained=cfg.PRETRAINED_DEVICE,
        ).to(device)
        encoder.load_state_dict(base_state, strict=True)
        encoder.eval()

        rho1, rho2, rho3 = rho_values
        ratio1 = rho_to_pruning_ratio(rho1)
        ratio2 = rho_to_pruning_ratio(rho2)
        ratio3 = rho_to_pruning_ratio(rho3)
        prune_unstructured(encoder.device_u1, ratio=ratio1, mode=cfg.PRUNE_MODE, seed=cfg.SEED + 1)
        prune_unstructured(encoder.device_u2, ratio=ratio2, mode=cfg.PRUNE_MODE, seed=cfg.SEED + 2)
        prune_unstructured(encoder.device_u3, ratio=ratio3, mode=cfg.PRUNE_MODE, seed=cfg.SEED + 3)

        predictions: List[str] = []
        references: List[List[str]] = []
        latency_sum = 0.0
        for batch in loader:
            start = time.perf_counter()
            with torch.no_grad():
                view_features = encoder(
                    batch["frames_u1"].to(device, non_blocking=True),
                    batch["frames_u2"].to(device, non_blocking=True),
                    batch["frames_u3"].to(device, non_blocking=True),
                ).cpu()
                view_features = apply_view_weights(view_features, sv_weights)
            preds = server_model.generate_from_view_features(
                view_features=view_features,
                prompt_texts=[cfg.USER_PROMPT],
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                num_beams=cfg.NUM_BEAMS,
            )
            latency_sum += time.perf_counter() - start
            predictions.extend(preds)
            references.extend(batch["references"])

        metrics = compute_caption_metrics(predictions, references)
        row = {
            "case_id": case_idx,
            "rho1": rho1,
            "rho2": rho2,
            "rho3": rho3,
            "rho_mean": (rho1 + rho2 + rho3) / 3.0,
            "pruning_ratio1": ratio1,
            "pruning_ratio2": ratio2,
            "pruning_ratio3": ratio3,
            "CIDEr": metrics["CIDEr"],
            "BLEU-4": metrics["BLEU-4"],
            "METEOR": metrics["METEOR"],
            "ROUGE-L": metrics["ROUGE-L"],
            "sparsity_u1": compute_sparsity(encoder.device_u1),
            "sparsity_u2": compute_sparsity(encoder.device_u2),
            "sparsity_u3": compute_sparsity(encoder.device_u3),
            "avg_latency_sec": latency_sum / max(1, len(predictions)),
            "upload_tokens": float(3 * cfg.DEVICE_TOKENS_PER_VIEW),
            "sv_weights_raw": format_float_list(sv_weights_raw),
            "sv_weights_applied": format_float_list(sv_weights.tolist()),
            "sv_scale_mode": args.sv_scale_mode,
        }
        rows.append(row)

        pred_file = out_dir / f"predictions_case_{case_idx:02d}_rho_{rho1:.2f}_{rho2:.2f}_{rho3:.2f}.json"
        pred_file.write_text(
            json.dumps(
                [
                    {
                        "prediction": prediction,
                        "references": refs,
                        "rho": rho_values,
                        "sv_weights_raw": sv_weights_raw,
                        "sv_weights_applied": sv_weights.tolist(),
                        "sv_scale_mode": args.sv_scale_mode,
                    }
                    for prediction, refs in zip(predictions, references)
                ],
                indent=2,
            )
        )
        print(
            f"[case {case_idx:02d}] rho=({format_float_list(rho_values)}) | "
            f"CIDEr={row['CIDEr']:.4f} | BLEU-4={row['BLEU-4']:.4f} | "
            f"sparsity=({row['sparsity_u1']*100:.2f}%, {row['sparsity_u2']*100:.2f}%, {row['sparsity_u3']*100:.2f}%)"
        )

    csv_path = out_dir / "pruning_sweep_caption.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if shared_rho_only:
        xs = [row["rho1"] for row in rows]
        plot_curve(xs, [row["CIDEr"] for row in rows], "CIDEr", out_dir / "cider_vs_rho.png")
        plot_curve(xs, [row["BLEU-4"] for row in rows], "BLEU-4", out_dir / "bleu4_vs_rho.png")
        plot_curve(xs, [row["sparsity_u1"] * 100 for row in rows], "Device sparsity (%)", out_dir / "sparsity_vs_rho.png")
        plot_curve(xs, [row["avg_latency_sec"] for row in rows], "Avg latency (sec)", out_dir / "latency_vs_rho.png")
    else:
        print("[Info] asymmetric rho triplets detected, skipped scalar rho plots to avoid misleading x-axis.")
    print(f"[Save] sweep csv -> {csv_path}")


if __name__ == "__main__":
    main()
