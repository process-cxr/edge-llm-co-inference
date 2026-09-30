from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    from .config import cfg, cfg_asdict, ensure_dir, resolve_path
    from .dataset_msrvtt import build_default_cached_dataset, collate_cached_feature_batch
    from .models import QwenCaptionModel, count_trainable_parameters, set_seed
except ImportError:
    from config import cfg, cfg_asdict, ensure_dir, resolve_path
    from dataset_msrvtt import build_default_cached_dataset, collate_cached_feature_batch
    from models import QwenCaptionModel, count_trainable_parameters, set_seed


def default_stage_dir(stage: str) -> Path:
    path = resolve_path(cfg.SAVE_DIR) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_projector_ckpt() -> Path:
    return default_stage_dir("projector") / cfg.PROJECTOR_CKPT_NAME


def default_adapter_dir() -> Path:
    return default_stage_dir("qlora") / cfg.QLORA_DIR_NAME


def build_cached_loaders() -> Tuple[DataLoader, DataLoader, int]:
    train_dataset = build_default_cached_dataset(
        split="train",
        unique_videos=False,
        max_samples=cfg.MAX_TRAIN_SAMPLES,
    )
    val_dataset = build_default_cached_dataset(
        split="val",
        unique_videos=False,
        max_samples=cfg.MAX_VAL_SAMPLES,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_cached_feature_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_cached_feature_batch,
    )
    sample = train_dataset[0]
    feature_dim = int(sample["view_features"].shape[-1])
    return train_loader, val_loader, feature_dim


def evaluate_loss(model: QwenCaptionModel, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                view_features=batch["view_features"],
                prompt_texts=[cfg.USER_PROMPT] * len(batch["caption_texts"]),
                target_texts=batch["caption_texts"],
            )
            total += float(outputs.loss.item()) * len(batch["caption_texts"])
            count += len(batch["caption_texts"])
    return total / max(1, count)


def save_training_artifacts(
    model: QwenCaptionModel,
    stage_dir: Path,
    stage: str,
    best_val_loss: float,
    history: Dict[str, object],
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    model.save_projector_checkpoint(
        stage_dir / cfg.PROJECTOR_CKPT_NAME,
        extra={"best_val_loss": best_val_loss, "stage": stage},
    )
    model.tokenizer.save_pretrained(stage_dir / "tokenizer")

    if stage == "qlora" and model.qlora_enabled:
        model.lm.save_pretrained(stage_dir / cfg.QLORA_DIR_NAME)

    with (stage_dir / "training_state.json").open("w") as file:
        json.dump(history, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the server-side Qwen caption model.")
    parser.add_argument("--stage", choices=["projector", "qlora"], required=True)
    parser.add_argument("--projector-ckpt", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    set_seed(cfg.SEED)
    ensure_dir(cfg.SAVE_DIR)
    train_loader, val_loader, feature_dim = build_cached_loaders()

    use_4bit = bool(cfg.USE_4BIT and torch.cuda.is_available())
    model = QwenCaptionModel(
        server_model_name=cfg.SERVER_MODEL_NAME,
        view_feature_dim=feature_dim,
        tokens_per_view=cfg.DEVICE_TOKENS_PER_VIEW,
        use_4bit=use_4bit,
    )

    if args.stage == "projector":
        model.train_projector_only()
        epochs = cfg.EPOCHS_PROJECTOR
        lr = cfg.LR_PROJECTOR
    else:
        projector_ckpt = Path(args.projector_ckpt) if args.projector_ckpt else default_projector_ckpt()
        if not projector_ckpt.exists():
            raise FileNotFoundError(
                f"Projector checkpoint required for qlora stage but not found: {projector_ckpt}"
            )
        model.load_projector_checkpoint(projector_ckpt)
        model.train_projector_only()
        if hasattr(model.lm, "gradient_checkpointing_enable"):
            model.lm.gradient_checkpointing_enable()
        model.enable_qlora(
            lora_r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES,
        )
        epochs = cfg.EPOCHS_QLORA
        lr = cfg.LR_QLORA

    output_dir = Path(args.output_dir) if args.output_dir else default_stage_dir(args.stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.json").open("w") as file:
        json.dump(cfg_asdict(), file, indent=2)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    best_val_loss = float("inf")
    history = {
        "stage": args.stage,
        "train_losses": [],
        "val_losses": [],
    }

    print(f"[Info] stage={args.stage} | trainable_params={count_trainable_parameters(model)/1e6:.2f}M")
    print(f"[Info] output_dir={output_dir}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            outputs = model(
                view_features=batch["view_features"],
                prompt_texts=[cfg.USER_PROMPT] * len(batch["caption_texts"]),
                target_texts=batch["caption_texts"],
            )
            loss = outputs.loss / cfg.GRAD_ACCUM_STEPS
            loss.backward()

            running += float(outputs.loss.item()) * len(batch["caption_texts"])
            seen += len(batch["caption_texts"])

            if step % cfg.GRAD_ACCUM_STEPS == 0:
                clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.EVAL_EVERY_STEPS == 0:
                val_loss = evaluate_loss(model, val_loader)
                avg_train_loss = running / max(1, seen)
                history["train_losses"].append({"step": global_step, "loss": avg_train_loss})
                history["val_losses"].append({"step": global_step, "loss": val_loss})
                print(
                    f"[Step {global_step:05d}] epoch={epoch}/{epochs} | "
                    f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_training_artifacts(model, output_dir, args.stage, best_val_loss, history)
                    print(f"[Save] best_val_loss={best_val_loss:.4f}")

        if len(train_loader) % cfg.GRAD_ACCUM_STEPS != 0:
            clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_loss = evaluate_loss(model, val_loader)
        avg_train_loss = running / max(1, seen)
        history["train_losses"].append({"epoch": epoch, "loss": avg_train_loss})
        history["val_losses"].append({"epoch": epoch, "loss": val_loss})
        print(
            f"[Epoch {epoch:02d}] train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_training_artifacts(model, output_dir, args.stage, best_val_loss, history)
            print(f"[Save] best_val_loss={best_val_loss:.4f}")

    print(f"[Done] stage={args.stage} | best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
