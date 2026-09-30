from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Dict, Tuple


THIS_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("LAIM_MSRVTT_DIR", str(THIS_DIR / "data" / "msrvtt")))
VIDEO_ROOT = Path(os.environ.get("LAIM_MSRVTT_VIDEO_DIR", str(DATA_ROOT / "videos" / "all")))
ANNOTATION_FILE = Path(os.environ.get("LAIM_MSRVTT_ANNOTATION", str(DATA_ROOT / "annotations" / "MSR_VTT.json")))


@dataclass
class CFG:
    # Paths
    DATA_DIR: str = str(DATA_ROOT)
    VIDEO_DIR: str = str(VIDEO_ROOT)
    ANNOTATION_PATH: str = str(ANNOTATION_FILE)
    CACHE_DIR: str = "../outputs/video_captioning/cache/msrvtt_frames"
    FEATURE_CACHE_DIR: str = "../outputs/video_captioning/cache/msrvtt_tinyvit_features"
    SAVE_DIR: str = "../outputs/video_captioning"
    PREDICTION_DIR: str = "../outputs/video_captioning/predictions"
    METRICS_DIR: str = "../outputs/video_captioning/metrics"

    # Data
    NUM_FRAMES: int = 8
    FRAME_SIZE: int = 224
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    USER_RANGES: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.2),
        (0.2, 0.6),
        (0.5, 1.0),
    )
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Models
    DEVICE_MODEL_NAME: str = "tiny_vit_5m_224.dist_in22k_ft_in1k"
    SERVER_MODEL_NAME: str = "Qwen/Qwen3-4B-Base"
    DEVICE_TOKENS_PER_VIEW: int = 1
    PRETRAINED_DEVICE: bool = True

    # Prompting / generation
    SYSTEM_PROMPT: str = "You are a concise video captioning assistant."
    USER_PROMPT: str = "Describe the video content from three partial visual views in one sentence."
    MAX_TARGET_TOKENS: int = 48
    MAX_NEW_TOKENS: int = 48
    NUM_BEAMS: int = 1

    # Train
    BATCH_SIZE: int = 2
    GRAD_ACCUM_STEPS: int = 8
    EPOCHS_PROJECTOR: int = 2
    EPOCHS_QLORA: int = 1
    LR_PROJECTOR: float = 5e-4
    LR_QLORA: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    EVAL_EVERY_STEPS: int = 100
    USE_4BIT: bool = True
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    MAX_TRAIN_SAMPLES: int = 0
    MAX_VAL_SAMPLES: int = 0
    MAX_TEST_SAMPLES: int = 0
    RANDOM_TEST_SUBSET: bool = False

    # Pruning
    PRUNE_MODE: str = "magnitude"
    RHO_LIST: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)

    # Reproducibility
    SEED: int = 42

    # Artifacts
    PROJECTOR_CKPT_NAME: str = "best_projector.pt"
    QLORA_DIR_NAME: str = "best_qlora"


cfg = CFG()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (THIS_DIR / path).resolve()


def ensure_dir(path_str: str) -> Path:
    path = resolve_path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cfg_asdict() -> Dict[str, object]:
    return asdict(cfg)
