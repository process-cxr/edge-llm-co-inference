from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from .config import cfg, resolve_path
except ImportError:
    from config import cfg, resolve_path

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - optional dependency
    VideoReader = None
    cpu = None

try:
    from torchvision.io import read_video as tv_read_video
except ImportError:  # pragma: no cover - torchvision build without video API
    tv_read_video = None


def _video_number(video_id: str) -> int:
    match = re.findall(r"\d+", video_id)
    if not match:
        raise ValueError(f"Cannot infer numeric index from video id: {video_id}")
    return int(match[-1])


def infer_default_split(video_id: str) -> str:
    idx = _video_number(video_id)
    if idx <= 6512:
        return "train"
    if idx <= 7009:
        return "val"
    return "test"


def normalize_caption_text(text: str) -> str:
    return " ".join(text.strip().split())


class MSRVTTIndex:
    """Parse MSR-VTT annotations and expose caption items plus unique videos."""

    VIDEO_SUFFIXES: Tuple[str, ...] = (".mp4", ".webm", ".avi", ".mov", ".mkv")
    VALID_SPLITS = {"train", "val", "test"}

    def __init__(self, annotation_path: str, video_dir: str):
        self.annotation_path = resolve_path(annotation_path)
        self.video_dir = resolve_path(video_dir)
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"MSR-VTT annotation file not found: {self.annotation_path}")

        data = json.loads(self.annotation_path.read_text())
        videos = data.get("videos", [])
        sentences = data.get("sentences", [])
        annotation_splits = [
            self._normalize_split(video.get("split"))
            for video in videos
            if self._normalize_split(video.get("split")) is not None
        ]
        unique_annotation_splits = set(annotation_splits)
        self.use_inferred_split = len(unique_annotation_splits) <= 1
        if self.use_inferred_split and videos:
            print(
                "[Warn] MSR-VTT annotation split metadata is missing or collapsed to a single split. "
                "Falling back to id-based train/val/test partition."
            )

        self.video_records: Dict[str, Dict[str, object]] = {}
        for video in videos:
            video_id = str(video.get("video_id"))
            split = self._resolve_split(video.get("split"), video_id)
            self.video_records[video_id] = {
                "video_id": video_id,
                "split": split,
                "category": video.get("category"),
                "video_path": str(video.get("video_path")) if video.get("video_path") else None,
            }

        self.video_to_captions: Dict[str, List[str]] = defaultdict(list)
        self.caption_items: List[Dict[str, str]] = []
        for sentence in sentences:
            video_id = str(sentence.get("video_id"))
            caption = sentence.get("caption") or sentence.get("sentence") or ""
            caption = normalize_caption_text(str(caption))
            if not caption:
                continue
            if video_id not in self.video_records:
                self.video_records[video_id] = {
                    "video_id": video_id,
                    "split": infer_default_split(video_id),
                    "category": None,
                    "video_path": None,
                }
            self.video_to_captions[video_id].append(caption)
            self.caption_items.append(
                {
                    "video_id": video_id,
                    "caption_text": caption,
                    "split": str(self.video_records[video_id]["split"]),
                }
            )

        self.unique_video_items: List[Dict[str, object]] = []
        for video_id, record in self.video_records.items():
            self.unique_video_items.append(
                {
                    "video_id": video_id,
                    "split": str(record["split"]),
                    "captions": list(self.video_to_captions.get(video_id, [])),
                }
            )
        self._video_path_cache: Dict[str, Path] = {}
        self._video_file_index: Optional[Dict[str, Path]] = None

    def iter_caption_items(self, split: str) -> List[Dict[str, object]]:
        return [item for item in self.caption_items if item["split"] == split]

    def iter_unique_videos(self, split: str) -> List[Dict[str, object]]:
        return [item for item in self.unique_video_items if item["split"] == split]

    def get_references(self, video_id: str) -> List[str]:
        return list(self.video_to_captions.get(video_id, []))

    def _normalize_split(self, split_value: object) -> Optional[str]:
        if split_value is None:
            return None
        split = str(split_value).strip().lower()
        if split in {"validation", "valid"}:
            split = "val"
        if split in self.VALID_SPLITS:
            return split
        return None

    def _resolve_split(self, split_value: object, video_id: str) -> str:
        normalized = self._normalize_split(split_value)
        if normalized is not None and not self.use_inferred_split:
            return normalized
        return infer_default_split(video_id)

    def _index_video_files(self) -> Dict[str, Path]:
        file_index: Dict[str, Path] = {}
        if not self.video_dir.exists():
            return file_index

        for path in self.video_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.VIDEO_SUFFIXES:
                continue
            file_index.setdefault(path.stem, path.resolve())
        return file_index

    def resolve_video_path(self, video_id: str) -> Path:
        cached = self._video_path_cache.get(video_id)
        if cached is not None:
            return cached

        record = self.video_records.get(video_id) or {}
        video_path = record.get("video_path")
        if video_path:
            path = Path(str(video_path))
            if not path.is_absolute():
                path = self.video_dir / path
            if path.exists():
                resolved = path.resolve()
                self._video_path_cache[video_id] = resolved
                return resolved

        for suffix in self.VIDEO_SUFFIXES:
            candidate = self.video_dir / f"{video_id}{suffix}"
            if candidate.exists():
                resolved = candidate.resolve()
                self._video_path_cache[video_id] = resolved
                return resolved

        for child_dir in self.video_dir.iterdir() if self.video_dir.exists() else []:
            if not child_dir.is_dir():
                continue
            for suffix in self.VIDEO_SUFFIXES:
                candidate = child_dir / f"{video_id}{suffix}"
                if candidate.exists():
                    resolved = candidate.resolve()
                    self._video_path_cache[video_id] = resolved
                    return resolved

        if self._video_file_index is None:
            self._video_file_index = self._index_video_files()
        indexed = self._video_file_index.get(video_id)
        if indexed is not None:
            self._video_path_cache[video_id] = indexed
            return indexed

        raise FileNotFoundError(
            f"Could not resolve video file for {video_id} under {self.video_dir}. "
            f"Checked direct paths, one-level child directories, and a recursive index."
        )


def load_or_sample_video_frames(
    video_path: Path,
    num_frames: int,
    frame_size: int,
) -> torch.Tensor:
    if VideoReader is not None:
        reader = VideoReader(str(video_path), ctx=cpu(0))
        total = len(reader)
        if total == 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        indices = torch.linspace(0, total - 1, steps=num_frames).round().long().tolist()
        frames = torch.from_numpy(reader.get_batch(indices).asnumpy())  # [T,H,W,C]
    elif tv_read_video is not None:
        frames, _, _ = tv_read_video(str(video_path), pts_unit="sec")
        if frames.numel() == 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        total = frames.shape[0]
        indices = torch.linspace(0, total - 1, steps=num_frames).round().long()
        frames = frames.index_select(0, indices)
    else:
        raise RuntimeError(
            "No video decoding backend is available. Install `decord` or use a torchvision build "
            "that exposes `torchvision.io.read_video`."
        )

    frames = frames.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    frames = TF.resize(
        frames.float() / 255.0,
        [frame_size, frame_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return (frames * 255.0).round().clamp(0, 255).to(torch.uint8)


def make_partial_view_video(
    frames_u8: torch.Tensor,
    left: float,
    right: float,
    fill_mean: Sequence[float],
    norm_mean: Sequence[float],
    norm_std: Sequence[float],
) -> torch.Tensor:
    frames = frames_u8.float() / 255.0  # [T,C,H,W]
    _, _, _, width = frames.shape
    x0 = int(round(left * width))
    x1 = int(round(right * width))
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    if x1 < x0:
        x0, x1 = x1, x0

    fill = torch.tensor(fill_mean, dtype=frames.dtype).view(1, 3, 1, 1)
    mean = torch.tensor(norm_mean, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, dtype=frames.dtype).view(1, 3, 1, 1)
    out = fill.expand_as(frames).clone()
    out[:, :, :, x0:x1] = frames[:, :, :, x0:x1]
    out = (out - mean) / std
    return out


def build_frame_cache_path(cache_dir: str, split: str, video_id: str, num_frames: int, frame_size: int) -> Path:
    root = resolve_path(cache_dir) / split
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{video_id}_t{num_frames}_s{frame_size}.pt"


def build_feature_cache_path(feature_cache_dir: str, split: str, video_id: str) -> Path:
    root = resolve_path(feature_cache_dir) / split
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{video_id}.pt"


class ThreeViewMSRVTTDataset(Dataset):
    """Return three-view normalized video clips plus caption text."""

    def __init__(
        self,
        annotation_path: str,
        video_dir: str,
        split: str,
        num_frames: int,
        frame_size: int,
        user_ranges: Tuple[Tuple[float, float], ...],
        frame_cache_dir: str,
        fill_mean: Sequence[float],
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
        unique_videos: bool = False,
        max_samples: int = 0,
    ):
        super().__init__()
        self.index = MSRVTTIndex(annotation_path=annotation_path, video_dir=video_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.user_ranges = user_ranges
        self.frame_cache_dir = frame_cache_dir
        self.fill_mean = fill_mean
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.unique_videos = unique_videos

        if unique_videos:
            items = self.index.iter_unique_videos(split)
        else:
            items = self.index.iter_caption_items(split)
        if max_samples > 0:
            items = items[:max_samples]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _load_or_cache_frames(self, video_id: str) -> torch.Tensor:
        cache_path = build_frame_cache_path(
            cache_dir=self.frame_cache_dir,
            split=self.split,
            video_id=video_id,
            num_frames=self.num_frames,
            frame_size=self.frame_size,
        )
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")

        video_path = self.index.resolve_video_path(video_id)
        frames_u8 = load_or_sample_video_frames(video_path, num_frames=self.num_frames, frame_size=self.frame_size)
        torch.save(frames_u8, cache_path)
        return frames_u8

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[idx]
        video_id = str(item["video_id"])
        frames_u8 = self._load_or_cache_frames(video_id)

        views = []
        for left, right in self.user_ranges:
            views.append(
                make_partial_view_video(
                    frames_u8=frames_u8,
                    left=left,
                    right=right,
                    fill_mean=self.fill_mean,
                    norm_mean=self.norm_mean,
                    norm_std=self.norm_std,
                )
            )

        captions = self.index.get_references(video_id)
        caption_text = str(item.get("caption_text") or (captions[0] if captions else ""))
        return {
            "video_id": video_id,
            "frames_u1": views[0],
            "frames_u2": views[1],
            "frames_u3": views[2],
            "caption_text": caption_text,
            "references": captions,
        }


class CachedFeatureMSRVTTDataset(Dataset):
    """Load cached TinyViT features and pair them with captions."""

    def __init__(
        self,
        annotation_path: str,
        video_dir: str,
        feature_cache_dir: str,
        split: str,
        unique_videos: bool = False,
        max_samples: int = 0,
    ):
        super().__init__()
        self.index = MSRVTTIndex(annotation_path=annotation_path, video_dir=video_dir)
        self.feature_cache_dir = feature_cache_dir
        self.split = split
        self.unique_videos = unique_videos

        if unique_videos:
            items = self.index.iter_unique_videos(split)
        else:
            items = self.index.iter_caption_items(split)
        if max_samples > 0:
            items = items[:max_samples]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[idx]
        video_id = str(item["video_id"])
        cache_path = build_feature_cache_path(self.feature_cache_dir, self.split, video_id)
        if not cache_path.exists():
            raise FileNotFoundError(f"Feature cache not found for {video_id}: {cache_path}")

        payload = torch.load(cache_path, map_location="cpu")
        references = self.index.get_references(video_id)
        return {
            "video_id": video_id,
            "view_features": payload["view_features"].float(),
            "caption_text": str(item.get("caption_text") or (references[0] if references else "")),
            "references": references,
        }


def collate_raw_video_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "video_ids": [item["video_id"] for item in batch],
        "frames_u1": torch.stack([item["frames_u1"] for item in batch], dim=0),
        "frames_u2": torch.stack([item["frames_u2"] for item in batch], dim=0),
        "frames_u3": torch.stack([item["frames_u3"] for item in batch], dim=0),
        "caption_texts": [item["caption_text"] for item in batch],
        "references": [item["references"] for item in batch],
    }


def collate_cached_feature_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "video_ids": [item["video_id"] for item in batch],
        "view_features": torch.stack([item["view_features"] for item in batch], dim=0),
        "caption_texts": [item["caption_text"] for item in batch],
        "references": [item["references"] for item in batch],
    }


def build_default_raw_dataset(split: str, unique_videos: bool = False, max_samples: int = 0) -> ThreeViewMSRVTTDataset:
    return ThreeViewMSRVTTDataset(
        annotation_path=cfg.ANNOTATION_PATH,
        video_dir=cfg.VIDEO_DIR,
        split=split,
        num_frames=cfg.NUM_FRAMES,
        frame_size=cfg.FRAME_SIZE,
        user_ranges=cfg.USER_RANGES,
        frame_cache_dir=cfg.CACHE_DIR,
        fill_mean=cfg.IMAGENET_MEAN,
        norm_mean=cfg.IMAGENET_MEAN,
        norm_std=cfg.IMAGENET_STD,
        unique_videos=unique_videos,
        max_samples=max_samples,
    )


def build_default_cached_dataset(split: str, unique_videos: bool = False, max_samples: int = 0) -> CachedFeatureMSRVTTDataset:
    return CachedFeatureMSRVTTDataset(
        annotation_path=cfg.ANNOTATION_PATH,
        video_dir=cfg.VIDEO_DIR,
        feature_cache_dir=cfg.FEATURE_CACHE_DIR,
        split=split,
        unique_videos=unique_videos,
        max_samples=max_samples,
    )
