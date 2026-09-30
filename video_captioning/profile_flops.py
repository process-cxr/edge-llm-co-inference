#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from safetensors import safe_open
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoTokenizer

try:
    from .config import cfg, ensure_dir, resolve_path
    from .models import ThreeDeviceTinyViTEncoder, TinyViTDeviceEncoder, VisualPrefixProjector
except ImportError:
    from config import cfg, ensure_dir, resolve_path
    from models import ThreeDeviceTinyViTEncoder, TinyViTDeviceEncoder, VisualPrefixProjector


def _count_params(module: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in module.parameters())


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_gflops(value: float) -> str:
    return f"{value / 1e9:.4f}"


def _parse_csv_floats(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _estimate_upload_bits(feature_dim: int, num_views: int, tokens_per_view: int, bits_per_value: int) -> Dict[str, int]:
    per_user_values = feature_dim * tokens_per_view
    per_cluster_values = per_user_values * num_views
    return {
        "per_user_bits": per_user_values * bits_per_value,
        "per_cluster_bits": per_cluster_values * bits_per_value,
    }


def _load_tokenizer(tokenizer_path: Path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)


def _count_prediction_tokens(tokenizer: AutoTokenizer, predictions_path: Path) -> Dict[str, object]:
    payload = json.loads(predictions_path.read_text())
    token_lengths = [len(tokenizer(item["prediction"], add_special_tokens=False).input_ids) for item in payload]
    latencies = [float(item["latency_sec"]) for item in payload if "latency_sec" in item]
    if not token_lengths:
        raise ValueError(f"No predictions found in {predictions_path}")
    return {
        "count": len(token_lengths),
        "token_lengths": token_lengths,
        "avg_tokens": statistics.mean(token_lengths),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "avg_latency_sec": statistics.mean(latencies) if latencies else None,
        "min_latency_sec": min(latencies) if latencies else None,
        "max_latency_sec": max(latencies) if latencies else None,
    }


def _infer_lm_shape_from_adapter(adapter_path: Path) -> Dict[str, int]:
    layer_indices: List[int] = []
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    q_proj_out_dim: Optional[int] = None
    kv_proj_out_dim: Optional[int] = None
    lora_params = 0

    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        tensor_names = list(handle.keys())
        for name in tensor_names:
            shape = tuple(handle.get_tensor(name).shape)
            lora_params += math.prod(shape)
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                layer_indices.append(int(match.group(1)))
            if name.endswith("self_attn.q_proj.lora_A.weight"):
                hidden_size = int(shape[1])
            elif name.endswith("self_attn.q_proj.lora_B.weight"):
                q_proj_out_dim = int(shape[0])
            elif name.endswith("self_attn.k_proj.lora_B.weight"):
                kv_proj_out_dim = int(shape[0])
            elif name.endswith("mlp.gate_proj.lora_B.weight"):
                intermediate_size = int(shape[0])

    if not layer_indices:
        raise ValueError(f"Could not infer transformer depth from {adapter_path}")
    if hidden_size is None or intermediate_size is None or q_proj_out_dim is None or kv_proj_out_dim is None:
        raise ValueError(f"Could not infer full LM shape from {adapter_path}")

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": max(layer_indices) + 1,
        "q_proj_out_dim": q_proj_out_dim,
        "kv_proj_out_dim": kv_proj_out_dim,
        "lora_params": lora_params,
    }


def _estimate_prefill_flops(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    q_proj_out_dim: int,
    kv_proj_out_dim: int,
    vocab_size: int,
    include_lm_head: bool,
) -> int:
    attn_and_mlp = (
        2 * seq_len * hidden_size * q_proj_out_dim
        + 2 * seq_len * hidden_size * kv_proj_out_dim
        + 2 * seq_len * hidden_size * kv_proj_out_dim
        + 2 * seq_len * q_proj_out_dim * hidden_size
        + 4 * seq_len * seq_len * q_proj_out_dim
        + 6 * seq_len * hidden_size * intermediate_size
    )
    flops = num_hidden_layers * attn_and_mlp
    if include_lm_head:
        flops += 2 * seq_len * hidden_size * vocab_size
    return int(flops)


def _estimate_decode_flops(
    prefix_tokens: int,
    generated_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    q_proj_out_dim: int,
    kv_proj_out_dim: int,
    vocab_size: int,
    include_lm_head: bool,
) -> int:
    total = 0
    for step in range(int(generated_tokens)):
        context_len = prefix_tokens + step
        attn_and_mlp = (
            2 * hidden_size * q_proj_out_dim
            + 2 * hidden_size * kv_proj_out_dim
            + 2 * hidden_size * kv_proj_out_dim
            + 2 * q_proj_out_dim * hidden_size
            + 4 * context_len * q_proj_out_dim
            + 6 * hidden_size * intermediate_size
        )
        total += num_hidden_layers * attn_and_mlp
        if include_lm_head:
            total += 2 * hidden_size * vocab_size
    return int(total)


@torch.inference_mode()
def measure_device_stack(
    model_name: str,
    num_frames: int,
    frame_size: int,
) -> Dict[str, int]:
    single_device = TinyViTDeviceEncoder(model_name=model_name, pretrained=False).eval()
    three_device = ThreeDeviceTinyViTEncoder(model_name=model_name, pretrained=False).eval()

    frames = torch.randn(1, num_frames, 3, frame_size, frame_size)
    with FlopCounterMode(display=False) as one_user_mode:
        one_user_output = single_device(frames)
    with FlopCounterMode(display=False) as three_user_mode:
        three_user_output = three_device(frames, frames, frames)

    return {
        "feature_dim": int(single_device.feature_dim),
        "single_user_flops": int(one_user_mode.get_total_flops()),
        "single_user_params": _count_params(single_device),
        "three_user_flops": int(three_user_mode.get_total_flops()),
        "three_user_params": _count_params(three_device),
        "single_user_output_dim": int(one_user_output.shape[-1]),
        "three_user_num_views": int(three_user_output.shape[1]),
    }


@torch.inference_mode()
def measure_projector_stack(input_dim: int, output_dim: int, tokens_per_view: int, num_views: int) -> Dict[str, int]:
    projector = VisualPrefixProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        tokens_per_view=tokens_per_view,
        num_views=num_views,
    ).eval()
    features = torch.randn(1, num_views, input_dim)
    with FlopCounterMode(display=False) as mode:
        output = projector(features)
    return {
        "projector_flops": int(mode.get_total_flops()),
        "projector_params": _count_params(projector),
        "projector_output_tokens": int(output.shape[1]),
        "projector_output_dim": int(output.shape[2]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile compute and upload costs for video captioning.")
    parser.add_argument("--device-model-name", default=cfg.DEVICE_MODEL_NAME)
    parser.add_argument("--num-frames", type=int, default=cfg.NUM_FRAMES)
    parser.add_argument("--frame-size", type=int, default=cfg.FRAME_SIZE)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--tokens-per-view", type=int, default=cfg.DEVICE_TOKENS_PER_VIEW)
    parser.add_argument("--server-model-name", default=cfg.SERVER_MODEL_NAME)
    parser.add_argument("--tokenizer-path", type=str, default=str(resolve_path("./outputs/qlora/tokenizer")))
    parser.add_argument("--adapter-path", type=str, default=str(resolve_path("./outputs/qlora/best_qlora/adapter_model.safetensors")))
    parser.add_argument("--prompt-text", type=str, default="")
    parser.add_argument("--predictions-file", type=str, default=str(resolve_path("./outputs/predictions/test_qlora.json")))
    parser.add_argument("--generated-tokens", type=int, default=cfg.MAX_NEW_TOKENS)
    parser.add_argument("--dtype-bits", type=str, default="16,32")
    parser.add_argument("--rho-list", type=str, default="1.0,0.9,0.8,0.7,0.6,0.5,0.4")
    parser.add_argument("--exclude-lm-head", action="store_true")
    parser.add_argument("--out-json", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    tokenizer_path = Path(args.tokenizer_path)
    adapter_path = Path(args.adapter_path)
    predictions_path = Path(args.predictions_file)
    rho_list = _parse_csv_floats(args.rho_list)
    include_lm_head = not args.exclude_lm_head

    device_report = measure_device_stack(
        model_name=args.device_model_name,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )
    lm_shape = _infer_lm_shape_from_adapter(adapter_path)

    prompt_text = args.prompt_text or (
        f"System: {cfg.SYSTEM_PROMPT}\nUser: {cfg.USER_PROMPT}\nAssistant:"
    )
    tokenizer = _load_tokenizer(tokenizer_path)
    prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=True).input_ids)
    vocab_size = int(tokenizer.vocab_size)
    prefix_tokens = prompt_tokens + args.num_views * args.tokens_per_view

    projector_report = measure_projector_stack(
        input_dim=device_report["feature_dim"],
        output_dim=lm_shape["hidden_size"],
        tokens_per_view=args.tokens_per_view,
        num_views=args.num_views,
    )

    dtype_bits_list = [int(value) for value in _parse_csv_floats(args.dtype_bits)]
    upload_report = {
        f"fp{bits}": _estimate_upload_bits(
            feature_dim=device_report["feature_dim"],
            num_views=args.num_views,
            tokens_per_view=args.tokens_per_view,
            bits_per_value=bits,
        )
        for bits in dtype_bits_list
    }

    theoretical_device_gflops = [
        {
            "rho": rho,
            "per_user_gflops_dense_scaled": device_report["single_user_flops"] * rho / 1e9,
            "per_cluster_gflops_dense_scaled": device_report["three_user_flops"] * rho / 1e9,
        }
        for rho in rho_list
    ]

    prefill_flops = _estimate_prefill_flops(
        seq_len=prefix_tokens,
        hidden_size=lm_shape["hidden_size"],
        intermediate_size=lm_shape["intermediate_size"],
        num_hidden_layers=lm_shape["num_hidden_layers"],
        q_proj_out_dim=lm_shape["q_proj_out_dim"],
        kv_proj_out_dim=lm_shape["kv_proj_out_dim"],
        vocab_size=vocab_size,
        include_lm_head=include_lm_head,
    )
    max_decode_flops = _estimate_decode_flops(
        prefix_tokens=prefix_tokens,
        generated_tokens=args.generated_tokens,
        hidden_size=lm_shape["hidden_size"],
        intermediate_size=lm_shape["intermediate_size"],
        num_hidden_layers=lm_shape["num_hidden_layers"],
        q_proj_out_dim=lm_shape["q_proj_out_dim"],
        kv_proj_out_dim=lm_shape["kv_proj_out_dim"],
        vocab_size=vocab_size,
        include_lm_head=include_lm_head,
    )

    predictions_report = None
    server_from_predictions = None
    if predictions_path.exists():
        predictions_report = _count_prediction_tokens(tokenizer=tokenizer, predictions_path=predictions_path)
        per_sample_server_flops = [
            prefill_flops
            + _estimate_decode_flops(
                prefix_tokens=prefix_tokens,
                generated_tokens=token_count,
                hidden_size=lm_shape["hidden_size"],
                intermediate_size=lm_shape["intermediate_size"],
                num_hidden_layers=lm_shape["num_hidden_layers"],
                q_proj_out_dim=lm_shape["q_proj_out_dim"],
                kv_proj_out_dim=lm_shape["kv_proj_out_dim"],
                vocab_size=vocab_size,
                include_lm_head=include_lm_head,
            )
            + projector_report["projector_flops"]
            for token_count in predictions_report["token_lengths"]
        ]
        sorted_server_flops = sorted(per_sample_server_flops)
        vector_indices = [round(i * (len(sorted_server_flops) - 1) / 7) for i in range(8)]
        server_from_predictions = {
            "avg_server_flops": int(statistics.mean(per_sample_server_flops)),
            "min_server_flops": int(min(per_sample_server_flops)),
            "max_server_flops": int(max(per_sample_server_flops)),
            "median_server_flops": int(statistics.median(per_sample_server_flops)),
            "suggested_server_gflops_vector_8clusters": [
                round(sorted_server_flops[idx] / 1e9, 3) for idx in vector_indices
            ],
        }

    report = {
        "device_model_name": args.device_model_name,
        "server_model_name": args.server_model_name,
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "num_views": args.num_views,
        "tokens_per_view": args.tokens_per_view,
        "prompt_tokens": prompt_tokens,
        "prefix_tokens": prefix_tokens,
        "device_report": device_report,
        "upload_report": upload_report,
        "projector_report": projector_report,
        "lm_shape": {
            key: value
            for key, value in lm_shape.items()
            if key != "lora_params"
        },
        "lora_params": lm_shape["lora_params"],
        "server_report": {
            "prefill_flops": prefill_flops + projector_report["projector_flops"],
            "decode_flops_max_new_tokens": max_decode_flops,
            "total_flops_max_new_tokens": prefill_flops + projector_report["projector_flops"] + max_decode_flops,
            "include_lm_head": include_lm_head,
        },
        "predictions_report": predictions_report,
        "server_from_predictions": server_from_predictions,
        "matlab_recommendation": {
            "Nflop_device_dense_per_user": device_report["single_user_flops"] / 1e9,
            "Nflop_device_dense_per_cluster": device_report["three_user_flops"] / 1e9,
            "Nflop_device_theory_scales_with_rho": theoretical_device_gflops,
            "theta_bits_per_user_fp16": upload_report.get("fp16", {}).get("per_user_bits"),
            "theta_bits_per_user_fp32": upload_report.get("fp32", {}).get("per_user_bits"),
            "theta_cluster_bits_fp16": upload_report.get("fp16", {}).get("per_cluster_bits"),
            "theta_cluster_bits_fp32": upload_report.get("fp32", {}).get("per_cluster_bits"),
            "Nflop_server_avg_from_predictions": (
                None if server_from_predictions is None else server_from_predictions["avg_server_flops"] / 1e9
            ),
            "Nflop_server_max_new_tokens": (
                prefill_flops + projector_report["projector_flops"] + max_decode_flops
            )
            / 1e9,
        },
    }

    print("=== Video Caption Split Profile ===")
    print(f"device_model                : {args.device_model_name}")
    print(f"server_model                : {args.server_model_name}")
    print(f"input                       : views={args.num_views}, frames/view={args.num_frames}, size={args.frame_size}x{args.frame_size}")
    print("")
    print(f"device params / user        : {_format_int(device_report['single_user_params'])}")
    print(f"device params / cluster     : {_format_int(device_report['three_user_params'])}")
    print(f"device FLOPs / user         : {_format_int(device_report['single_user_flops'])}  ({_format_gflops(device_report['single_user_flops'])} GFLOPs)")
    print(f"device FLOPs / cluster      : {_format_int(device_report['three_user_flops'])}  ({_format_gflops(device_report['three_user_flops'])} GFLOPs)")
    print("")
    print(f"feature_dim                 : {device_report['feature_dim']}")
    for dtype_key, stats in upload_report.items():
        print(
            f"upload {dtype_key} / user        : {_format_int(stats['per_user_bits'])} bits  "
            f"({stats['per_user_bits'] / 1e6:.6f} Mbits)"
        )
        print(
            f"upload {dtype_key} / cluster     : {_format_int(stats['per_cluster_bits'])} bits  "
            f"({stats['per_cluster_bits'] / 1e6:.6f} Mbits)"
        )
    print("")
    print(f"projector params            : {_format_int(projector_report['projector_params'])}")
    print(f"projector FLOPs             : {_format_int(projector_report['projector_flops'])}  ({_format_gflops(projector_report['projector_flops'])} GFLOPs)")
    print("")
    print(
        "server LM shape             : "
        f"layers={lm_shape['num_hidden_layers']}, hidden={lm_shape['hidden_size']}, "
        f"intermediate={lm_shape['intermediate_size']}, q_out={lm_shape['q_proj_out_dim']}, "
        f"kv_out={lm_shape['kv_proj_out_dim']}, vocab={vocab_size}"
    )
    print(f"LoRA params                 : {_format_int(lm_shape['lora_params'])}")
    print(f"prompt tokens               : {prompt_tokens}")
    print(f"prefix tokens               : {prefix_tokens}  [visual + prompt]")
    print(f"server prefill FLOPs        : {_format_int(prefill_flops + projector_report['projector_flops'])}  ({_format_gflops(prefill_flops + projector_report['projector_flops'])} GFLOPs)")
    print(f"server decode FLOPs max     : {_format_int(max_decode_flops)}  ({_format_gflops(max_decode_flops)} GFLOPs)  [generated_tokens={args.generated_tokens}]")
    print(
        f"server total FLOPs max      : {_format_int(prefill_flops + projector_report['projector_flops'] + max_decode_flops)}  "
        f"({_format_gflops(prefill_flops + projector_report['projector_flops'] + max_decode_flops)} GFLOPs)"
    )
    print("")
    print("MATLAB mapping:")
    print(f"- Nflop_device dense/user   ~ {device_report['single_user_flops'] / 1e9:.4f}")
    print(f"- Nflop_device dense/cluster~ {device_report['three_user_flops'] / 1e9:.4f}")
    if upload_report.get("fp32"):
        print(f"- theta (fp32 per user)     ~ {upload_report['fp32']['per_user_bits']}")
    if upload_report.get("fp16"):
        print(f"- theta (fp16 per user)     ~ {upload_report['fp16']['per_user_bits']}")
    if server_from_predictions is not None:
        avg_server = server_from_predictions["avg_server_flops"]
        print(f"- Nflop_server avg/task     ~ {avg_server / 1e9:.4f}")
        print(
            "- Nflop_server 8-task vec   ~ "
            f"{server_from_predictions['suggested_server_gflops_vector_8clusters']}"
        )
    print("- rho scaling note          : current pruning is unstructured zeroing; dense runtime may not shrink linearly with rho.")

    out_json = Path(args.out_json) if args.out_json else ensure_dir(cfg.METRICS_DIR) / "profile_flops_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print("")
    print(f"[Save] report -> {out_json}")


if __name__ == "__main__":
    main()
