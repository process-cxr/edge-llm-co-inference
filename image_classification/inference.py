import os
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
from torchvision import transforms

from transformers import ViTForImageClassification


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = THIS_DIR.parent / "outputs" / "image_classification"
DATA_ROOT = Path(os.environ.get("LAIM_CIFAR10_DATA", str(THIS_DIR / "data")))



@dataclass
class CFG:
    """Experiment configuration for pruning-time evaluation."""
    # -------- Paths --------
    DATA_DIR: str = str(DATA_ROOT)
    CACHE_DIR: str = str(OUTPUT_ROOT / "cache_cifar10_224_u8")
    CKPT_PATH: str = str(THIS_DIR / "checkpoints" / "best.pt")

    # -------- Data --------
    IMAGE_SIZE: int = 224
    NUM_WORKERS: int = 8
    BATCH_SIZE: int = 64
    PIN_MEMORY: bool = True

    # Evaluate on subset of test set
    NUM_TEST_SAMPLES: int = 200
    RANDOM_TEST_SUBSET: bool = False


    USER_RANGES: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.2),  # user1
        (0.2, 0.6),  # user2
        (0.5, 1.0),  # user3
    )

# Equal importance
    # USER_RANGES: Tuple[Tuple[float, float], ...] = (
    #     (0.0, 0.3),  # user1
    #     (0.3, 0.6),  # user2
    #     (0.6, 1.0),  # user3
    # )

    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


    MODEL_NAME: str = "WinKawaks/vit-tiny-patch16-224"
    NUM_CLASSES: int = 10
    SPLIT_LAYERS: int = 3



    PRUNE_MODE: str = "magnitude" #  "random" or "magnitude"

    # 与 MATLAB `main_new_gemini.m` 保持一致：rho 表示保留比例 / 计算比例。
    # 实际传给 PyTorch 非结构化剪枝函数时，会做 pruning_ratio = 1 - rho 的转换。
    RHO1: float = 1.0
    RHO2: float = 1.0
    RHO3: float = 1.0

    # 兼容旧写法：若显式设置了 PRUNING_RATIO*，则优先按旧语义读取，
    # 再自动换算为 rho = 1 - pruning_ratio。
    PRUNING_RATIO1: Optional[float] = None
    PRUNING_RATIO2: Optional[float] = None
    PRUNING_RATIO3: Optional[float] = None


    SEED: int = 42


    USE_AMP: bool = True


cfg = CFG()



def set_seed(seed: int) -> None:
    """Keep python/torch RNG deterministic for repeatable pruning masks."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def rho_to_pruning_ratio(rho: float) -> float:
    """Convert MATLAB-style rho to PyTorch pruning ratio via pruning_ratio = 1 - rho."""
    assert 0.0 <= rho <= 1.0, f"rho must be in [0,1], got {rho}"
    return 1.0 - rho



def resolve_rho_and_pruning_ratio(rho: Optional[float], pruning_ratio: Optional[float], name: str) -> Tuple[float, float]:
    """Prefer rho semantics, but keep PRUNING_RATIO* as a deprecated compatibility alias."""
    if pruning_ratio is not None:
        assert 0.0 <= pruning_ratio <= 1.0, f"{name} pruning_ratio must be in [0,1], got {pruning_ratio}"
        rho = 1.0 - pruning_ratio

    if rho is None:
        raise ValueError(f"{name} must provide either rho or pruning_ratio")

    assert 0.0 <= rho <= 1.0, f"{name} rho must be in [0,1], got {rho}"
    return rho, rho_to_pruning_ratio(rho)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()



class MultiUserPartialCIFAR10Cached(Dataset):
    """Build three partial views of each CIFAR10 image and cache resized tensors."""

    def __init__(self,
                 root: str,
                 train: bool,
                 download: bool,
                 image_size: int,
                 user_ranges: Tuple[Tuple[float, float], ...],
                 cache_dir: str,
                 fill_mean: Tuple[float, float, float],
                 norm_mean: Tuple[float, float, float],
                 norm_std: Tuple[float, float, float]):
        super().__init__()
        self.base = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.user_ranges = user_ranges
        self.image_size = image_size

        self.fill_mean = torch.tensor(fill_mean).view(3, 1, 1)
        self.normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

        self.resize_only = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.to_tensor = transforms.ToTensor()

        split = "train" if train else "test"
        self.cache_root = os.path.join(cache_dir, split)
        os.makedirs(self.cache_root, exist_ok=True)

    def __len__(self) -> int:
        return len(self.base)

    def _cache_path(self, idx: int) -> str:
        return os.path.join(self.cache_root, f"{idx:06d}.pt")

    def _load_or_build_full224_uint8(self, idx: int) -> torch.Tensor:
        """Resize once to 224x224, store uint8 on disk to avoid recompute."""
        path = self._cache_path(idx)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")

        pil_img, _ = self.base[idx]
        pil_img = self.resize_only(pil_img)
        t = self.to_tensor(pil_img)
        u8 = (t * 255.0).clamp(0, 255).to(torch.uint8)
        torch.save(u8, path)
        return u8

    def _make_partial_view(self, full_u8: torch.Tensor, left: float, right: float) -> torch.Tensor:
        """Mask out everything outside [left, right) fraction of width, then normalize."""
        full = full_u8.float() / 255.0
        _, H, W = full.shape
        x0 = int(round(left * W))
        x1 = int(round(right * W))
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        if x1 < x0:
            x0, x1 = x1, x0

        out = self.fill_mean.expand_as(full).clone()
        out[:, :, x0:x1] = full[:, :, x0:x1]
        out = self.normalize(out)
        return out

    def __getitem__(self, idx: int):
        _, label = self.base[idx]
        full_u8 = self._load_or_build_full224_uint8(idx)

        v = []
        for (l, r) in self.user_ranges:
            v.append(self._make_partial_view(full_u8, l, r))

        return v[0], v[1], v[2], label


class ViTDevicePart(nn.Module):
    """Embedding + first N transformer blocks that live on each device."""
    def __init__(self, embeddings: nn.Module, blocks: nn.ModuleList):
        super().__init__()
        self.embeddings = embeddings
        self.blocks = blocks

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        for blk in self.blocks:
            hidden_states = blk(hidden_states)
        return hidden_states


class ViTServerPart(nn.Module):
    """remaining encoder blocks + final layernorm + classifier"""
    def __init__(self, blocks: nn.ModuleList, layernorm: nn.Module, classifier: nn.Module):
        super().__init__()
        self.blocks = blocks
        self.layernorm = layernorm
        self.classifier = classifier

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            hidden_states = blk(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        cls = hidden_states[:, 0]
        logits = self.classifier(cls)
        return logits


class ThreeDeviceSplitViTMeanFusion(nn.Module):
    """Three parallel device branches + mean fusion + shared server tail."""

    def __init__(self, model_name: str, num_classes: int, split_layers: int):
        super().__init__()

        base = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        vit = base.vit
        all_blocks = vit.encoder.layer
        num_blocks = len(all_blocks)
        assert 0 <= split_layers <= num_blocks, f"split_layers must be in [0,{num_blocks}], got {split_layers}"

        self.split_layers = split_layers
        self.num_blocks = num_blocks

        def build_one_device() -> ViTDevicePart:
            emb = copy.deepcopy(vit.embeddings)
            first_blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(split_layers)])
            return ViTDevicePart(embeddings=emb, blocks=first_blocks)

        self.device_u1 = build_one_device()
        self.device_u2 = build_one_device()
        self.device_u3 = build_one_device()

        server_blocks = nn.ModuleList([all_blocks[i] for i in range(split_layers, num_blocks)])
        self.server = ViTServerPart(
            blocks=server_blocks,
            layernorm=vit.layernorm,
            classifier=base.classifier,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        z1 = self.device_u1(x1)
        z2 = self.device_u2(x2)
        z3 = self.device_u3(x3)

        z = (z1 + z2 + z3) / 3.0
        logits = self.server(z)
        return logits


def _collect_prunable_weight_tensors(module: nn.Module) -> List[torch.Tensor]:
    """Return all weight tensors (ndim>=2) that we allow unstructured pruning on."""

    prunable = []
    for name, p in module.named_parameters():
        if ("weight" in name) and (p.data.ndim >= 2):
            prunable.append(p)
    return prunable


@torch.no_grad()
def compute_sparsity(module: nn.Module) -> float:
    params = _collect_prunable_weight_tensors(module)
    if not params:
        return 0.0
    total = 0
    zeros = 0
    for p in params:
        t = p.data
        total += t.numel()
        zeros += (t == 0).sum().item()
    return zeros / max(1, total)


@torch.no_grad()
def prune_unstructured(module: nn.Module,
                       ratio: float,
                       mode: str,
                       seed: int = 0) -> None:
    """
    Zero out a given ratio of weights across all prunable tensors.
    mode:
        - "random": pick weights uniformly at random
        - "magnitude": prune the smallest |weight| entries globally
    """

    assert 0.0 <= ratio <= 1.0, f"ratio must be in [0,1], got {ratio}"
    mode = mode.lower()
    assert mode in ["random", "magnitude"], f"mode must be 'random' or 'magnitude', got {mode}"

    params = _collect_prunable_weight_tensors(module)
    if not params:
        return

    device = params[0].data.device
    flats = [p.data.view(-1) for p in params]
    sizes = [f.numel() for f in flats]
    total = int(sum(sizes))

    k = int(total * ratio)
    if k <= 0:
        return
    if k >= total:

        for f in flats:
            f.zero_()
        return


    if mode == "random":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        idx = torch.randperm(total, generator=g, device=device)[:k]
    else:
        abs_all = torch.cat([f.abs() for f in flats], dim=0)
        _, idx = torch.topk(abs_all, k, largest=False)

    idx = torch.sort(idx).values

    offset = 0
    for f, n in zip(flats, sizes):
        end = offset + n

        # find idx in [offset, end)
        off_t = torch.tensor(offset, device=device)
        end_t = torch.tensor(end, device=device)
        left = torch.searchsorted(idx, off_t, right=False)
        right = torch.searchsorted(idx, end_t, right=False)

        if right > left:
            local = idx[left:right] - offset
            f[local] = 0

        offset = end


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             use_amp: bool) -> Tuple[float, float]:
    """Run a full test pass and return (loss, top1 accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x1, x2, x3, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        x3 = x3.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x1, x2, x3)
                loss = F.cross_entropy(logits, y)
        else:
            logits = model(x1, x2, x3)
            loss = F.cross_entropy(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / max(1, n), total_acc / max(1, n)



def main():
    """Entry point: load checkpoint, optionally prune device parts, evaluate."""
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # --- Build test dataset ---
    test_ds = MultiUserPartialCIFAR10Cached(
        root=cfg.DATA_DIR,
        train=False,
        download=True,
        image_size=cfg.IMAGE_SIZE,
        user_ranges=cfg.USER_RANGES,
        cache_dir=cfg.CACHE_DIR,
        fill_mean=cfg.IMAGENET_MEAN,
        norm_mean=cfg.IMAGENET_MEAN,
        norm_std=cfg.IMAGENET_STD,
    )

    total_test = len(test_ds)
    n_use = min(cfg.NUM_TEST_SAMPLES, total_test)
    indices = list(range(total_test))
    if cfg.RANDOM_TEST_SUBSET:
        rng = random.Random(cfg.SEED)
        rng.shuffle(indices)
    indices = indices[:n_use]
    test_subset = Subset(test_ds, indices)

    test_loader = DataLoader(
        test_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        drop_last=False,
    )

    model = ThreeDeviceSplitViTMeanFusion(
        model_name=cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        split_layers=cfg.SPLIT_LAYERS,
    ).to(device)

    assert os.path.exists(cfg.CKPT_PATH), f"Checkpoint not found: {cfg.CKPT_PATH}"
    ckpt = torch.load(cfg.CKPT_PATH, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)} (show up to 10): {missing[:10]}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)} (show up to 10): {unexpected[:10]}")
    print(f"[Info] Loaded checkpoint: {cfg.CKPT_PATH}")


    loss0, acc0 = evaluate(model, test_loader, device, cfg.USE_AMP)
    print(f"[Before pruning]  test_samples={n_use} | loss={loss0:.4f} | acc={acc0*100:.2f}%")


    rho1, r1 = resolve_rho_and_pruning_ratio(cfg.RHO1, cfg.PRUNING_RATIO1, "u1")
    rho2, r2 = resolve_rho_and_pruning_ratio(cfg.RHO2, cfg.PRUNING_RATIO2, "u2")
    rho3, r3 = resolve_rho_and_pruning_ratio(cfg.RHO3, cfg.PRUNING_RATIO3, "u3")
    if any(value is not None for value in (cfg.PRUNING_RATIO1, cfg.PRUNING_RATIO2, cfg.PRUNING_RATIO3)):
        print("[Warn] PRUNING_RATIO* is deprecated; converted internally via rho = 1 - pruning_ratio.")
    print(
        f"[Pruning] mode={cfg.PRUNE_MODE} | "
        f"rho=(u1={rho1}, u2={rho2}, u3={rho3}) | "
        f"pruning_ratio=(u1={r1}, u2={r2}, u3={r3}) | server: no pruning"
    )


    s1_b = compute_sparsity(model.device_u1)
    s2_b = compute_sparsity(model.device_u2)
    s3_b = compute_sparsity(model.device_u3)
    print(f"[Device sparsity before] u1={s1_b*100:.2f}% | u2={s2_b*100:.2f}% | u3={s3_b*100:.2f}%")


    prune_unstructured(model.device_u1, ratio=r1, mode=cfg.PRUNE_MODE, seed=cfg.SEED + 1)
    prune_unstructured(model.device_u2, ratio=r2, mode=cfg.PRUNE_MODE, seed=cfg.SEED + 2)
    prune_unstructured(model.device_u3, ratio=r3, mode=cfg.PRUNE_MODE, seed=cfg.SEED + 3)

    s1_a = compute_sparsity(model.device_u1)
    s2_a = compute_sparsity(model.device_u2)
    s3_a = compute_sparsity(model.device_u3)
    print(f"[Device sparsity after ] u1={s1_a*100:.2f}% | u2={s2_a*100:.2f}% | u3={s3_a*100:.2f}%")


    loss1, acc1 = evaluate(model, test_loader, device, cfg.USE_AMP)
    print(f"[After pruning ]  test_samples={n_use} | loss={loss1:.4f} | acc={acc1*100:.2f}%")

    print(f"[Delta] acc: {acc1*100:.2f}% - {acc0*100:.2f}% = {(acc1-acc0)*100:.2f}% points")


if __name__ == "__main__":
    main()
