import os
import time
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = THIS_DIR.parent / "outputs" / "image_classification"
DATA_ROOT = Path(os.environ.get("LAIM_CIFAR10_DATA", str(THIS_DIR / "data")))


@dataclass
class CFG:
    """Experiment config for end-to-end split ViT training on partial CIFAR10 views."""
    # Paths
    DATA_DIR: str = str(DATA_ROOT)
    CACHE_DIR: str = str(OUTPUT_ROOT / "cache_cifar10_224_u8")
    SAVE_DIR: str = str(OUTPUT_ROOT / "checkpoints")


    NUM_WORKERS: int = 8
    BATCH_SIZE: int = 64
    PIN_MEMORY: bool = True


    IMAGE_SIZE: int = 224


    USER_RANGES: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.2),  # user1
        (0.2, 0.6),  # user2
        (0.5, 1.0),  # user3
    )


    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


    MODEL_NAME: str = "WinKawaks/vit-tiny-patch16-224"
    NUM_CLASSES: int = 10
    SPLIT_LAYERS: int = 3

    TRAIN_END2END: bool = True


    EPOCHS: int = 10
    LR_HEAD: float = 5e-4
    LR_BACKBONE: float = 5e-5
    WEIGHT_DECAY: float = 0.05
    WARMUP_RATIO: float = 0.05


    USE_AMP: bool = True
    GRAD_CLIP_NORM: float = 1.0


    TRAIN_RANDOM_HFLIP: bool = True
    HFLIP_P: float = 0.5


    LOG_EVERY_STEPS: int = 50
    EVAL_FULL_TEST: bool = False
    EVAL_NUM_BATCHES: int = 50


    SEED: int = 42


    SAVE_BEST: bool = True


cfg = CFG()



def set_seed(seed: int) -> None:
    """Keep RNG deterministic across python/torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_curves(save_dir: str,
                steps: List[int],
                loss_curve: List[float],
                acc_curve: List[float]) -> None:
    """Persist loss/accuracy curves for quick inspection."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(steps, loss_curve)
    plt.xlabel("Global step")
    plt.ylabel("Train loss (avg over window)")
    plt.title("Training Loss vs Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_train_loss.png"), dpi=200)
    plt.close()


    plt.figure()
    plt.plot(steps, acc_curve)
    plt.xlabel("Global step")
    plt.ylabel("Test accuracy")
    plt.title("Test Accuracy vs Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_test_accuracy.png"), dpi=200)
    plt.close()


    fig, ax1 = plt.subplots()
    ax1.plot(steps, loss_curve)
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("Train loss (avg over window)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(steps, acc_curve)
    ax2.set_ylabel("Test accuracy")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "curve_loss_and_accuracy.png"), dpi=200)
    plt.close(fig)




class MultiUserPartialCIFAR10Cached(Dataset):
    """Create three masked partial views per CIFAR10 image and cache resized tensors."""
    """Create three masked partial views per CIFAR10 image and cache resized tensors."""
    def __init__(self,
                 root: str,
                 train: bool,
                 download: bool,
                 image_size: int,
                 user_ranges: Tuple[Tuple[float, float], ...],
                 cache_dir: str,
                 fill_mean: Tuple[float, float, float],
                 norm_mean: Tuple[float, float, float],
                 norm_std: Tuple[float, float, float],
                 random_hflip: bool = True,
                 hflip_p: float = 0.5):
        super().__init__()
        self.base = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.train = train
        self.user_ranges = user_ranges
        self.image_size = image_size

        self.fill_mean = torch.tensor(fill_mean).view(3, 1, 1)
        self.normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

        # Deterministic resize used for caching
        self.resize_only = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.to_tensor = transforms.ToTensor()

        split = "train" if train else "test"
        self.cache_root = os.path.join(cache_dir, split)
        os.makedirs(self.cache_root, exist_ok=True)

        self.random_hflip = random_hflip and train
        self.hflip_p = hflip_p

    def __len__(self) -> int:
        return len(self.base)

    def _cache_path(self, idx: int) -> str:
        return os.path.join(self.cache_root, f"{idx:06d}.pt")

    def _load_or_build_full224_uint8(self, idx: int) -> torch.Tensor:
        """Deterministically resize once to 224x224 and store uint8 on disk."""
        path = self._cache_path(idx)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")

        pil_img, _ = self.base[idx]
        pil_img = self.resize_only(pil_img)
        t = self.to_tensor(pil_img)  # float [0,1]
        u8 = (t * 255.0).clamp(0, 255).to(torch.uint8)
        torch.save(u8, path)
        return u8

    def _maybe_hflip(self, full_u8: torch.Tensor) -> torch.Tensor:
        if self.random_hflip and (random.random() < self.hflip_p):
            return torch.flip(full_u8, dims=[2])  # flip W
        return full_u8

    def _make_partial_view(self, full_u8: torch.Tensor, left: float, right: float) -> torch.Tensor:
        """Keep only horizontal slice [left,right); fill rest with channel mean and normalize."""
        full = full_u8.float() / 255.0  # [0,1]
        _, H, W = full.shape
        x0 = int(round(left * W))
        x1 = int(round(right * W))
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        if x1 < x0:
            x0, x1 = x1, x0

        out = self.fill_mean.expand_as(full).clone()
        out[:, :, x0:x1] = full[:, :, x0:x1]
        out = self.normalize(out)  # missing region becomes 0
        return out

    def __getitem__(self, idx: int):
        _, label = self.base[idx]
        full_u8 = self._load_or_build_full224_uint8(idx)
        full_u8 = self._maybe_hflip(full_u8)

        v = []
        for (l, r) in self.user_ranges:
            v.append(self._make_partial_view(full_u8, l, r))

        return v[0], v[1], v[2], label




class ViTDevicePart(nn.Module):
    """Embedding + first N transformer blocks that reside on each device."""
    def __init__(self, embeddings: nn.Module, blocks: nn.ModuleList):
        super().__init__()
        self.embeddings = embeddings
        self.blocks = blocks

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)  # (B,N,D)
        for blk in self.blocks:
            hidden_states = blk(hidden_states)
        return hidden_states


class ViTServerPart(nn.Module):
    """Remaining encoder blocks + final LayerNorm + classifier on server."""
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
    """Three device branches (shared weights) -> mean fusion -> shared server tail."""
    def __init__(self, model_name: str, num_classes: int, split_layers: int):
        super().__init__()

        base = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # classifier resized to num_classes
        )
        vit = base.vit
        all_blocks = vit.encoder.layer
        num_blocks = len(all_blocks)
        assert 0 <= split_layers <= num_blocks, f"split_layers must be in [0,{num_blocks}], got {split_layers}"

        self.split_layers = split_layers
        self.num_blocks = num_blocks


        def build_one_device_from_base() -> ViTDevicePart:
            emb = copy.deepcopy(vit.embeddings)
            first_blocks = nn.ModuleList([copy.deepcopy(all_blocks[i]) for i in range(split_layers)])
            return ViTDevicePart(embeddings=emb, blocks=first_blocks)

        self.device_u1 = build_one_device_from_base()
        self.device_u2 = build_one_device_from_base()
        self.device_u3 = build_one_device_from_base()


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


def build_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "server.classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    return torch.optim.AdamW(param_groups)




@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             use_amp: bool,
             max_batches: Optional[int] = None) -> Tuple[float, float]:
    """Eval loop returning (loss, top1 accuracy). Optionally cap batches for speed."""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for bi, (x1, x2, x3, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

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
    """Train split ViT end-to-end (3 device branches + server) on partial CIFAR10."""
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    set_seed(cfg.SEED)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")


    train_ds = MultiUserPartialCIFAR10Cached(
        root=cfg.DATA_DIR,
        train=True,
        download=True,
        image_size=cfg.IMAGE_SIZE,
        user_ranges=cfg.USER_RANGES,
        cache_dir=cfg.CACHE_DIR,
        fill_mean=cfg.IMAGENET_MEAN,
        norm_mean=cfg.IMAGENET_MEAN,
        norm_std=cfg.IMAGENET_STD,
        random_hflip=cfg.TRAIN_RANDOM_HFLIP,
        hflip_p=cfg.HFLIP_P,
    )
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
        random_hflip=False,
        hflip_p=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
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

    for p in model.parameters():
        p.requires_grad = True

    optimizer = build_optimizer(
        model,
        lr_head=cfg.LR_HEAD,
        lr_backbone=cfg.LR_BACKBONE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    total_steps = cfg.EPOCHS * len(train_loader)
    warmup_steps = int(cfg.WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Mode: END2END (3 independent devices + shared server)")
    print(f"[Info] split_layers(device)={cfg.SPLIT_LAYERS}, server={model.num_blocks-cfg.SPLIT_LAYERS}, total_blocks={model.num_blocks}")
    print(f"[Info] Trainable params: {n_trainable/1e6:.2f}M")
    print(f"[Info] cache_dir={cfg.CACHE_DIR} (full 224x224 uint8)")
    print(f"[Info] log_every_steps={cfg.LOG_EVERY_STEPS}, eval_full_test={cfg.EVAL_FULL_TEST}")

    best_test_acc = 0.0
    global_step = 0


    log_steps: List[int] = []
    log_train_loss: List[float] = []
    log_test_acc: List[float] = []

    loss_window_sum = 0.0
    loss_window_cnt = 0

    t_start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        for x1, x2, x3, y in train_loader:
            global_step += 1

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if cfg.USE_AMP and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x1, x2, x3)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x1, x2, x3)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                optimizer.step()

            scheduler.step()

            loss_window_sum += loss.item()
            loss_window_cnt += 1

            if global_step % cfg.LOG_EVERY_STEPS == 0:
                avg_loss = loss_window_sum / max(1, loss_window_cnt)
                loss_window_sum, loss_window_cnt = 0.0, 0

                if cfg.EVAL_FULL_TEST:
                    _, test_acc = evaluate(model, test_loader, device, cfg.USE_AMP, max_batches=None)
                else:
                    _, test_acc = evaluate(model, test_loader, device, cfg.USE_AMP, max_batches=cfg.EVAL_NUM_BATCHES)

                log_steps.append(global_step)
                log_train_loss.append(avg_loss)
                log_test_acc.append(test_acc)

                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                print(f"[Step {global_step:06d}] epoch={epoch}/{cfg.EPOCHS} | "
                      f"lr={lr_now:.2e} | avg_train_loss={avg_loss:.4f} | test_acc={test_acc*100:.2f}% | "
                      f"elapsed={elapsed/60:.1f} min")

                save_curves(cfg.SAVE_DIR, log_steps, log_train_loss, log_test_acc)

                if cfg.SAVE_BEST and test_acc > best_test_acc:
                    best_test_acc = test_acc
                    ckpt_path = os.path.join(cfg.SAVE_DIR, "best.pt")
                    torch.save({
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_test_acc": best_test_acc,
                        "model_state": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "log_steps": log_steps,
                        "log_train_loss": log_train_loss,
                        "log_test_acc": log_test_acc,
                    }, ckpt_path)
                    print(f"[Save] Best checkpoint: {ckpt_path} (best_test_acc={best_test_acc*100:.2f}%)")

    if log_steps:
        save_curves(cfg.SAVE_DIR, log_steps, log_train_loss, log_test_acc)

    print(f"[Done] Best test acc: {best_test_acc*100:.2f}%")
    print(f"[Done] Curves saved in: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()
