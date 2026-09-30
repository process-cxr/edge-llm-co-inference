#!/usr/bin/env python3
# profile_flops.py --device cuda --batch-size 1 --image-size 224 --split-layers 3
import argparse
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode


def _count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


@dataclass
class FlopsReport:
    params_total: int
    params_trainable: int
    flops_total: int
    flops_device_one_user: int
    flops_server: int


@torch.inference_mode()
def measure_split_vit_flops(
    model_name: str,
    num_classes: int,
    split_layers: int,
    image_size: int,
    batch_size: int,
    device: str,
) -> FlopsReport:
    """
    Measure FLOPs for the split-ViT forward used in this repo.

    Notes:
    - FLOPs are input-shape dependent. Use batch_size=1 for per-sample FLOPs.
    - PyTorch's FlopCounterMode counts ops it can see at the aten-op level.
      Some fused kernels may be under-counted depending on backend.
    """
    # Import lazily to keep CLI startup fast and avoid side effects unless needed.
    from end2end import ThreeDeviceSplitViTMeanFusion

    torch_device = torch.device(device)
    model = ThreeDeviceSplitViTMeanFusion(
        model_name=model_name,
        num_classes=num_classes,
        split_layers=split_layers,
    ).to(torch_device)
    model.eval()

    x1 = torch.randn(batch_size, 3, image_size, image_size, device=torch_device)
    x2 = torch.randn(batch_size, 3, image_size, image_size, device=torch_device)
    x3 = torch.randn(batch_size, 3, image_size, image_size, device=torch_device)

    params_total = _count_params(model)
    params_trainable = _count_trainable_params(model)

    # Full model FLOPs
    with FlopCounterMode(display=False) as mode_total:
        _ = model(x1, x2, x3)
    flops_total = int(mode_total.get_total_flops())

    # One-device FLOPs (per user)
    with FlopCounterMode(display=False) as mode_dev:
        _ = model.device_u1(x1)
    flops_device_one_user = int(mode_dev.get_total_flops())

    # Server FLOPs (per fused hidden state)
    z = model.device_u1(x1)
    with FlopCounterMode(display=False) as mode_srv:
        _ = model.server(z)
    flops_server = int(mode_srv.get_total_flops())

    return FlopsReport(
        params_total=params_total,
        params_trainable=params_trainable,
        flops_total=flops_total,
        flops_device_one_user=flops_device_one_user,
        flops_server=flops_server,
    )


def _format_int(n: int) -> str:
    return f"{n:,}"


def _format_gflops(n_flops: int) -> str:
    return f"{(n_flops / 1e9):.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure FLOPs for split ViT model in this repo.")
    p.add_argument("--model-name", default="WinKawaks/vit-tiny-patch16-224")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--split-layers", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rep = measure_split_vit_flops(
        model_name=args.model_name,
        num_classes=args.num_classes,
        split_layers=args.split_layers,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Heuristic decomposition: total ~= 3*device_one_user + server (+ small fusion ops).
    device_total_3_users = 3 * rep.flops_device_one_user
    residual = rep.flops_total - (device_total_3_users + rep.flops_server)

    print("=== Split ViT FLOPs Report ===")
    print(f"model_name     : {args.model_name}")
    print(f"split_layers   : {args.split_layers}")
    print(f"input          : batch={args.batch_size}, image={args.image_size}x{args.image_size}")
    print(f"device         : {args.device}")
    print("")
    print(f"params_total   : {_format_int(rep.params_total)}")
    print(f"params_train   : {_format_int(rep.params_trainable)}")
    print("")
    print(f"FLOPs total    : {_format_int(rep.flops_total)}  ({_format_gflops(rep.flops_total)} GFLOPs)")
    print(
        f"FLOPs device/u : {_format_int(rep.flops_device_one_user)}  "
        f"({ _format_gflops(rep.flops_device_one_user)} GFLOPs)  [per user]"
    )
    print(f"FLOPs device(3): {_format_int(device_total_3_users)}  ({_format_gflops(device_total_3_users)} GFLOPs)")
    print(f"FLOPs server   : {_format_int(rep.flops_server)}  ({_format_gflops(rep.flops_server)} GFLOPs)")
    print(f"residual       : {_format_int(residual)}  ({_format_gflops(residual)} GFLOPs)  [fusion/overhead]")
    print("")
    print("For MATLAB optimization:")
    print(f"- Per-user device GFLOPs  ~ { _format_gflops(rep.flops_device_one_user) }")
    print(f"- Per-task server GFLOPs  ~ { _format_gflops(rep.flops_server) }")


if __name__ == "__main__":
    main()
