# Core simulation code for the LAIM co-inference paper

This directory is an independent, source-focused copy of the programs used
for the experiments in `paper/LAIM_co_inference_NPJ/LAIM_schedul/TWC20260605.tex`.
The original `vit_all/` tree is intentionally left unchanged.

> **Repository note.** This is a source-only snapshot. External datasets
> (CIFAR-10, MSR-VTT) and pretrained/model weights are **not** included.
> Several paths referenced below are not bundled and must be provided locally
> before running: the shared `common/` package (e.g. `common/shapley_core.py`
> and `common/shapley_outputs/`), the task `checkpoints/`, and the MATLAB
> `system_simulation/optimization/` and `system_simulation/mint/` solvers.
> See `SOURCE_MANIFEST.md` for the intended source-to-target mapping.

## Layout

- `common/`: exact three-device Shapley implementation and the measured
  per-cluster weights consumed by MATLAB.
- `image_classification/`: split ViT/CIFAR-10 training, inference, pruning,
  FLOPs and Shapley entry points. The bundled `checkpoints/best.pt` is the
  paper checkpoint.
- `video_captioning/`: three-view MSR-VTT, TinyViT/Qwen, projector/QLoRA,
  caption metrics, pruning and Shapley entry points. The bundled checkpoint
  directory contains the trained projector, adapter and tokenizer.
- `system_simulation/optimization/`: current configurable MATLAB solver and
  fixed-power, fixed-bandwidth, round-robin and random baselines.
- `system_simulation/mint/`: delay-timeline and bandwidth/pruning analysis.

## Python environments

Each task has its own `pyproject.toml` and lock file. Run commands from the
corresponding task directory, for example:

```bash
cd core_code/image_classification
uv sync
uv run python inference.py --device cpu
uv run python pic/plot_accuracy_threshold_curves.py
```

The CIFAR-10 dataset is external. Set `LAIM_CIFAR10_DATA` to its root before
training, inference or Shapley evaluation. The default checkpoint is the
bundled `image_classification/checkpoints/best.pt`.

For video experiments, set `LAIM_MSRVTT_DIR` to a directory containing
`annotations/MSR_VTT.json` and `videos/all/`. `LAIM_MSRVTT_VIDEO_DIR` and
`LAIM_MSRVTT_ANNOTATION` can override those two paths independently:

```bash
cd core_code/video_captioning
uv sync
export LAIM_MSRVTT_DIR=/path/to/msrvtt
uv run python inference_caption.py --input-mode raw --split test --num-samples 32
uv run python evaluate_caption.py --predictions ../outputs/video_captioning/predictions/test_predictions.json
```

The video server defaults to the bundled `checkpoints/qlora` artifacts and
falls back to newly trained artifacts under `outputs/video_captioning/`.

## Shapley weights

The measured weights are already available in `common/shapley_outputs/`.
To recompute them, run `compute_cluster_shapley.py` from the relevant task
directory after providing the external dataset/model dependencies. Both
scripts write to the shared output directory by default.

## MATLAB simulations

MATLAB with `intlinprog` and CVX is required. From MATLAB, add both folders
and run the desired entry point:

```matlab
addpath('core_code/system_simulation/optimization');
addpath('core_code/system_simulation/mint');
run('core_code/system_simulation/optimization/main_new_gemini.m');
run('core_code/system_simulation/mint/main_current_delay.m');
run('core_code/system_simulation/mint/main_current_bandwidth_insight.m');
```

The public single-case interface is
`run_case_main_new_gemini(run_cfg)`. `run_cfg` must include an 8-by-3
`importance_weights` matrix whose rows sum to 1.1; task profiles `vit` and
`video` and schedule modes `joint`, `fixed_p`, `round_robin` and
`random_scheduling` are supported. MATLAB writes generated figures and
summaries below `outputs/` and never modifies the paper directory.

The full source-to-target mapping is recorded in `SOURCE_MANIFEST.md`.
