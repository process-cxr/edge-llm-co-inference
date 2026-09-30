from __future__ import annotations

import math
import os
import random
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .config import cfg, resolve_path
except ImportError:
    from config import cfg, resolve_path

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on transformers build
    BitsAndBytesConfig = None

try:
    import timm
except ImportError:  # pragma: no cover - handled at runtime
    timm = None


def normalize_proxy_env() -> None:
    """Make proxy env vars compatible with httpx/huggingface_hub."""
    socks_supported = find_spec("socksio") is not None
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if not value:
            continue

        if value.startswith("socks://"):
            value = "socks5://" + value[len("socks://") :]
            os.environ[key] = value

        if value.startswith("socks5://") and not socks_supported:
            # Fall back to HTTP(S)_PROXY when socks support is unavailable.
            os.environ.pop(key, None)


normalize_proxy_env()


def _hf_cache_repo_dir(model_id: str) -> Path:
    repo_key = model_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_key}"


def _latest_valid_snapshot(repo_dir: Path) -> Optional[Path]:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir() and (path / "config.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_local_model_source(model_id: str) -> Optional[Path]:
    repo_dir = _hf_cache_repo_dir(model_id)
    return _latest_valid_snapshot(repo_dir)


def _resolve_local_tokenizer_source(model_id: str) -> Optional[Path]:
    candidate_dirs = [
        resolve_path("./outputs/qlora/tokenizer"),
        resolve_path("./outputs/projector/tokenizer"),
    ]
    for path in candidate_dirs:
        if (path / "tokenizer.json").exists():
            return path

    snapshot = _resolve_local_model_source(model_id)
    if snapshot is not None and (snapshot / "tokenizer.json").exists():
        return snapshot
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rho_to_pruning_ratio(rho: float) -> float:
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0,1], got {rho}")
    return 1.0 - rho


def _collect_prunable_weight_tensors(module: nn.Module) -> List[torch.Tensor]:
    prunable: List[torch.Tensor] = []
    for name, param in module.named_parameters():
        if "weight" in name and param.data.ndim >= 2:
            prunable.append(param)
    return prunable


@torch.no_grad()
def compute_sparsity(module: nn.Module) -> float:
    params = _collect_prunable_weight_tensors(module)
    if not params:
        return 0.0
    total = 0
    zeros = 0
    for param in params:
        tensor = param.data
        total += tensor.numel()
        zeros += (tensor == 0).sum().item()
    return zeros / max(1, total)


@torch.no_grad()
def prune_unstructured(module: nn.Module, ratio: float, mode: str, seed: int = 0) -> None:
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio must be in [0,1], got {ratio}")
    mode = mode.lower()
    if mode not in {"random", "magnitude"}:
        raise ValueError(f"Unsupported prune mode: {mode}")

    params = _collect_prunable_weight_tensors(module)
    if not params:
        return

    device = params[0].data.device
    flats = [param.data.view(-1) for param in params]
    sizes = [flat.numel() for flat in flats]
    total = int(sum(sizes))
    k = int(total * ratio)
    if k <= 0:
        return
    if k >= total:
        for flat in flats:
            flat.zero_()
        return

    if mode == "random":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = torch.randperm(total, generator=generator, device=device)[:k]
    else:
        abs_all = torch.cat([flat.abs() for flat in flats], dim=0)
        _, indices = torch.topk(abs_all, k, largest=False)

    indices = torch.sort(indices).values
    offset = 0
    for flat, size in zip(flats, sizes):
        end = offset + size
        left = torch.searchsorted(indices, torch.tensor(offset, device=device), right=False)
        right = torch.searchsorted(indices, torch.tensor(end, device=device), right=False)
        if right > left:
            local = indices[left:right] - offset
            flat[local] = 0
        offset = end


class TinyViTDeviceEncoder(nn.Module):
    """Encode a single view video clip into a temporally pooled feature."""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required to build TinyViT encoders.")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feature_dim = int(getattr(self.backbone, "num_features", 0))
        if self.feature_dim <= 0:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, cfg.FRAME_SIZE, cfg.FRAME_SIZE)
                self.feature_dim = int(self.backbone(dummy).shape[-1])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, num_frames, channels, height, width = frames.shape
        flat = frames.view(batch * num_frames, channels, height, width)
        features = self.backbone(flat)
        if features.ndim > 2:
            dims = tuple(range(2, features.ndim))
            features = features.mean(dim=dims)
        features = features.view(batch, num_frames, -1).mean(dim=1)
        return features


class ThreeDeviceTinyViTEncoder(nn.Module):
    """Three independent TinyViT branches, one per partial visual view."""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.device_u1 = TinyViTDeviceEncoder(model_name=model_name, pretrained=pretrained)
        self.device_u2 = TinyViTDeviceEncoder(model_name=model_name, pretrained=pretrained)
        self.device_u3 = TinyViTDeviceEncoder(model_name=model_name, pretrained=pretrained)
        self.feature_dim = self.device_u1.feature_dim

    def forward(self, frames_u1: torch.Tensor, frames_u2: torch.Tensor, frames_u3: torch.Tensor) -> torch.Tensor:
        z1 = self.device_u1(frames_u1)
        z2 = self.device_u2(frames_u2)
        z3 = self.device_u3(frames_u3)
        return torch.stack([z1, z2, z3], dim=1)  # [B, 3, D]


class VisualPrefixProjector(nn.Module):
    """Add device identity and project view features into the Qwen hidden space."""

    def __init__(self, input_dim: int, output_dim: int, tokens_per_view: int = 1, num_views: int = 3):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.num_views = num_views
        self.device_embedding = nn.Parameter(torch.zeros(num_views, input_dim))
        nn.init.normal_(self.device_embedding, std=0.02)

        self.token_expander: Optional[nn.Linear]
        if tokens_per_view > 1:
            self.token_expander = nn.Linear(input_dim, input_dim * tokens_per_view)
        else:
            self.token_expander = None

        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    @property
    def num_visual_tokens(self) -> int:
        return self.num_views * self.tokens_per_view

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        batch, num_views, dim = view_features.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {num_views}")

        x = view_features + self.device_embedding.unsqueeze(0)
        if self.token_expander is not None:
            x = self.token_expander(x)
            x = x.view(batch, self.num_visual_tokens, dim)
        else:
            x = x.view(batch, num_views, dim)
        return self.projector(x)


class QwenCaptionModel(nn.Module):
    """Server-side language model wrapper with trainable visual prefix projector."""

    def __init__(
        self,
        server_model_name: str,
        view_feature_dim: int,
        tokens_per_view: int = 1,
        use_4bit: bool = True,
    ):
        super().__init__()
        self.server_model_name = server_model_name
        self.use_4bit = use_4bit and torch.cuda.is_available()
        tokenizer_source = _resolve_local_tokenizer_source(server_model_name)
        model_source = _resolve_local_model_source(server_model_name)
        tokenizer_ref = str(tokenizer_source) if tokenizer_source is not None else server_model_name
        model_ref = str(model_source) if model_source is not None else server_model_name
        tokenizer_local_only = tokenizer_source is not None
        model_local_only = model_source is not None

        print(
            "[Info] Qwen tokenizer source: "
            f"{tokenizer_ref} {'(local)' if tokenizer_local_only else '(remote)'}"
        )
        print(
            "[Info] Qwen model source: "
            f"{model_ref} {'(local cache)' if model_local_only else '(remote)'}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_ref,
            use_fast=True,
            local_files_only=tokenizer_local_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        device_map = None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if self.use_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes + BitsAndBytesConfig are required for 4-bit loading.")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            device_map = {"": 0}

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_ref,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=model_local_only,
        )
        self.hidden_size = int(self.lm.config.hidden_size)
        self.projector = VisualPrefixProjector(
            input_dim=view_feature_dim,
            output_dim=self.hidden_size,
            tokens_per_view=tokens_per_view,
            num_views=3,
        )
        if not self.use_4bit:
            self.lm.to(self.model_device)
        self.projector.to(self.model_device)
        self.qlora_enabled = False

    @property
    def model_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    @property
    def projector_dtype(self) -> torch.dtype:
        return next(self.projector.parameters()).dtype

    @property
    def lm_embed_dtype(self) -> torch.dtype:
        embed_layer = self.lm.get_input_embeddings()
        weight = getattr(embed_layer, "weight", None)
        if weight is not None:
            return weight.dtype
        lm_head = getattr(self.lm, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight.dtype
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def freeze_lm(self) -> None:
        for param in self.lm.parameters():
            param.requires_grad = False

    def train_projector_only(self) -> None:
        self.freeze_lm()
        for param in self.projector.parameters():
            param.requires_grad = True

    def enable_qlora(self, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: Sequence[str]) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.use_4bit:
            self.lm = prepare_model_for_kbit_training(self.lm)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.lm = get_peft_model(self.lm, config)
        self.qlora_enabled = True
        for param in self.projector.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def _build_prompt(self, prompt_text: Optional[str] = None) -> str:
        user_prompt = prompt_text or cfg.USER_PROMPT
        return f"System: {cfg.SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"

    def _embed_inputs(
        self,
        projected_views: torch.Tensor,
        prompt_texts: Sequence[str],
        target_texts: Optional[Sequence[str]] = None,
        max_target_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = projected_views.size(0)
        embed_layer = self.lm.get_input_embeddings()
        projected_views = projected_views.to(dtype=self.lm_embed_dtype)
        pad_token_id = int(self.tokenizer.pad_token_id)
        eos_token_id = int(self.tokenizer.eos_token_id)
        pad_embed = embed_layer(torch.tensor([pad_token_id], device=self.model_device)).squeeze(0)

        sample_embeds: List[torch.Tensor] = []
        sample_masks: List[torch.Tensor] = []
        sample_labels: List[torch.Tensor] = []
        prompt_lengths: List[int] = []

        target_limit = max_target_tokens or cfg.MAX_TARGET_TOKENS

        for idx in range(batch):
            prompt = self._build_prompt(prompt_texts[idx] if idx < len(prompt_texts) else None)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True).input_ids
            text_ids = list(prompt_ids)
            labels = [-100] * (projected_views.size(1) + len(prompt_ids))

            if target_texts is not None:
                target = self.tokenizer(
                    str(target_texts[idx]),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=target_limit,
                ).input_ids
                target = target + [eos_token_id]
                text_ids.extend(target)
                labels.extend(target)

            text_tensor = torch.tensor(text_ids, device=self.model_device)
            text_embeds = embed_layer(text_tensor)
            merged = torch.cat([projected_views[idx], text_embeds], dim=0)

            sample_embeds.append(merged)
            sample_masks.append(torch.ones(merged.size(0), dtype=torch.long, device=self.model_device))
            sample_labels.append(torch.tensor(labels, dtype=torch.long, device=self.model_device))
            prompt_lengths.append(len(prompt_ids))

        max_len = max(item.size(0) for item in sample_embeds)
        padded_embeds = []
        padded_masks = []
        padded_labels = []
        for embeds, mask, labels in zip(sample_embeds, sample_masks, sample_labels):
            pad_len = max_len - embeds.size(0)
            if pad_len > 0:
                embeds = torch.cat([embeds, pad_embed.expand(pad_len, -1)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long, device=self.model_device)], dim=0)
                labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=torch.long, device=self.model_device)],
                    dim=0,
                )
            padded_embeds.append(embeds)
            padded_masks.append(mask)
            padded_labels.append(labels)

        batch_embeds = torch.stack(padded_embeds, dim=0)
        batch_masks = torch.stack(padded_masks, dim=0)
        batch_labels = torch.stack(padded_labels, dim=0)
        return {
            "inputs_embeds": batch_embeds,
            "attention_mask": batch_masks,
            "labels": batch_labels,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        }

    def forward(
        self,
        view_features: torch.Tensor,
        prompt_texts: Sequence[str],
        target_texts: Sequence[str],
        max_target_tokens: Optional[int] = None,
    ):
        view_features = view_features.to(device=self.model_device, dtype=self.projector_dtype)
        projected = self.projector(view_features).to(dtype=self.lm_embed_dtype)
        packed = self._embed_inputs(projected, prompt_texts=prompt_texts, target_texts=target_texts, max_target_tokens=max_target_tokens)
        return self.lm(
            inputs_embeds=packed["inputs_embeds"],
            attention_mask=packed["attention_mask"],
            labels=packed["labels"],
        )

    @torch.no_grad()
    def generate_from_view_features(
        self,
        view_features: torch.Tensor,
        prompt_texts: Sequence[str],
        max_new_tokens: int = 48,
        num_beams: int = 1,
    ) -> List[str]:
        self.eval()
        view_features = view_features.to(device=self.model_device, dtype=self.projector_dtype)
        projected = self.projector(view_features).to(dtype=self.lm_embed_dtype)

        generations: List[str] = []
        for idx in range(view_features.size(0)):
            prompt = self._build_prompt(prompt_texts[idx] if idx < len(prompt_texts) else None)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(self.model_device)
            prompt_embeds = self.lm.get_input_embeddings()(prompt_ids).squeeze(0)
            inputs_embeds = torch.cat([projected[idx], prompt_embeds], dim=0).unsqueeze(0)
            attention_mask = torch.ones((1, inputs_embeds.size(1)), dtype=torch.long, device=self.model_device)
            output_ids = self.lm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            generations.append(text)
        return generations

    def save_projector_checkpoint(self, path: Path, extra: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "projector_state": self.projector.state_dict(),
            "server_model_name": self.server_model_name,
            "tokens_per_view": self.projector.tokens_per_view,
            "extra": extra or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_projector_checkpoint(self, path: Path) -> Dict[str, object]:
        payload = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(payload["projector_state"], strict=True)
        return payload.get("extra", {})


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
