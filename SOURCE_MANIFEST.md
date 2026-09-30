# Source manifest

All entries below were copied from the existing working tree without moving
or deleting the originals. Paths are relative to the repository root.

| Destination | Original source | Role |
| --- | --- | --- |
| `common/shapley_core.py` | `vit_all/shapley_core.py` | Exact Shapley, alpha normalization and writers |
| `common/shapley_outputs/` | `vit_all/shapley_outputs/` | Paper-measured ViT/video coalition and weight inputs |
| `image_classification/*.py` | `vit_all/vit/*.py` (current entry points) | ViT training, inference, pruning, FLOPs and Shapley |
| `image_classification/pic/plot_accuracy_threshold_curves.py` | `vit_all/vit/pic/plot_accuracy_threshold_curves.py` | ViT threshold plots |
| `image_classification/checkpoints/best.pt` | `vit_all/vit/checkpoints_split_vit_cifar10/best.pt` | Bundled ViT checkpoint |
| `video_captioning/*.py` | `vit_all/video_caption_split/*.py` (current entry points) | Video data/model/training/evaluation/pruning/Shapley |
| `video_captioning/pic/plot_accuracy_threshold_curves.py` | `vit_all/video_caption_split/pic/plot_accuracy_threshold_curves.py` | Video threshold plots |
| `video_captioning/checkpoints/qlora/` | `vit_all/video_caption_split/outputs/qlora/` | Projector, LoRA adapter and tokenizer |
| `system_simulation/optimization/` | Current files from `vit_all/optimization/` and `vit_all/video_caption_split/optimization/` | Joint solver and baselines |
| `system_simulation/mint/` | Current analysis files from `vit_all/mint/` | Delay and bandwidth figure generation |
|                                                              |                                                              |                                                      |

Historical duplicate optimization programs, virtual environments, raw
datasets, frame/feature caches, generated figure images and development logs
are deliberately excluded.
