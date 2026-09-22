# learn-minimind

从零手写一个小型语言模型（MiniMind 风格），用于学习 LLM 的内部结构。

## 当前进度

- [x] 模型配置 `MiniMindConfig`（Llama 风格超参）
- [x] `RMSNorm` 归一化层
- [x] 旋转位置编码（RoPE）
- [x] 注意力层（GQA）
- [x] 前馈网络（FFN）
- [x] 完整模型组装
- [x] 数据集与预训练脚本
- [ ] MoE / 微调 / 推理对话脚本

## 环境要求

- Python 与 PyTorch（具体版本见 `pyproject.toml`）

## 安装

```bash
# 推荐用 uv（仓库里的 uv.lock 会锁定依赖版本）
uv sync
```

## 项目结构

```
learn_minimind/
├── model/
│   └── model.py            # 模型定义（MiniMindConfig / RMSNorm / Attention / ...）
├── dataset/
│   └── im_dataset.py       # 预训练数据集封装
├── trainer/
│   ├── train_pretrain.py   # 预训练入口
│   └── trainer_utils.py    # 训练工具（学习率、日志、检查点、DDP）
├── out/                    # 训练产出的权重
├── checkpoints/            # 续训用的完整训练状态
└── pyproject.toml
```

## 训练

在项目根目录执行：

```bash
python trainer/train_pretrain.py
```

数据放在 `dataset/pretrain_hq.jsonl`，每行一个 `{"text": "..."}`。
续训加 `--from_resume 1`，基于已有权重继续训练加 `--from_weight pretrain`。


## 查看训练结果

在项目根目录执行：

```bash
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -u eval.py --weight pretrain_v3 --preset knowledge --max_new_tokens 40
```

## 许可证

本项目使用 [MIT License](LICENSE)。
