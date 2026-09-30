# LLM训练推理使用指南

## end2end.py

直接训练较小模型，输出 checkpoints best.pt 文件，以供 inference.py 推理使用。

```
uv run python end2end.py
```



## finetune.py

微调训练较大模型部分层，输出 checkpoints best.pt 文件，以供 inference.py 推理使用。

```
uv run python finetune.py
```



## inference.py

测试模型推理性能和剪枝后性能对比。

```
uv run python inference.py
```



## profile_flops.py

实测模型 FLOPs

```
uv run python profile_flops.py --device cuda --batch-size 1 --image-size 224 --split-layers 3
```



## sweep_pruning.py

扫描剪枝率并画折线图

```
uv run python sweep_pruning.py --start 0 --end 1 --step 0.05 --device cuda
```





## 注意：

- 模型训练的准确度不是重点，随剪枝率非线性变化较大的区域越大才是金指标
