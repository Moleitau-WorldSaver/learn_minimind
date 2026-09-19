# learn-minimind

从零手写一个小型语言模型（MiniMind 风格），用于学习 LLM 的内部结构。

> ⚠️ 项目正在开发中：目前实现了模型配置与 RMSNorm，其余部分逐步补齐。

## 当前进度

- [x] 模型配置 `MyMindConfig`（Llama 风格超参 + 可选 MoE）
- [x] `RMSNorm` 归一化层
- [x] 旋转位置编码（RoPE）
- [x] 注意力层（GQA）
- [x] 前馈网络（FFN / MoE）
- [x] 完整模型组装
- [ ] 数据集与预训练脚本
- [ ] 推理与对话脚本

## 环境要求

- Python 与 PyTorch（具体版本见 `pyproject.toml`）

## 安装

```bash
# 推荐用 uv（仓库里的 uv.lock 会锁定依赖版本）
uv sync
```

或者用 pip：

```bash
pip install -e .
```

## 项目结构

```
learn_minimind/
├── model/
│   └── model.py            # 模型定义（MyMindConfig / RMSNorm / ...）
├── trainer/
│   ├── train_pretrain.py   # 预训练入口
│   └── trainer_utils.py    # 训练工具
├── dataset/
│   └── im_dataset.py       # 数据集封装
├── src/learn_minimind/
│   └── __init__.py
└── pyproject.toml
```

## 许可证

本项目使用 [MIT License](LICENSE)。
