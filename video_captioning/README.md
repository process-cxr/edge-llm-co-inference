# Video Caption Split Co-inference

This experiment package extends the image split-inference prototype to
three-view video captioning with the following default setup:

- Dataset: `MSR-VTT`
- Device encoders: three `TinyViT-5M` branches
- Server decoder: `Qwen3-4B-Base`
- Training path: cache TinyViT features first, then train the server-side
  projector and QLoRA adapters

## Expected data layout

The default configuration expects the following local layout:

```text
video_captioning/
├── data/
│   └── msrvtt/
│       ├── annotations/
│       │   └── MSR_VTT.json
│       └── videos/
│           └── all/
│               ├── video0.mp4
│               ├── video1.mp4
│               └── ...
```

The annotation loader supports the common MSR-VTT JSON structure with
`videos` and `sentences` arrays. If split labels are missing, the standard
video-id ranges are used:

- `video0` - `video6512`: `train`
- `video6513` - `video7009`: `val`
- `video7010` - `video9999`: `test`

## Install

From the repo root:

```bash
cd core_code/video_captioning
uv sync
```

## Default workflow

1. Cache per-video TinyViT features

```bash
uv run cache_features.py --split all
```

2. Train the projector only

```bash
uv run finetune_server.py --stage projector
```

3. Train QLoRA adapters on top of the frozen TinyViT features

```bash
uv run finetune_server.py --stage qlora
```

4. Generate captions from raw videos

```bash
uv run inference_caption.py --input-mode raw --split test --num-samples 32
```

5. Score the generated captions

```bash
uv run evaluate_caption.py --predictions outputs/predictions/test_predictions.json
```

6. Sweep device pruning ratios

```bash
uv run sweep_pruning_caption.py --split test --num-samples 32
```

## Notes

- The three-view spatial slicing follows the image prototype exactly:
  `(0.0, 0.2)`, `(0.2, 0.6)`, `(0.5, 1.0)`.
- The first version keeps the device pruning semantics aligned with the
  existing image scripts by using global unstructured magnitude pruning.
- The generated metrics include `CIDEr`, `BLEU-4`, `METEOR`, and `ROUGE-L`,
  with a local `CIDEr-lite` fallback that does not depend on external
  evaluation servers.
