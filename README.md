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
- [x] MoE（4 专家 + top-1 路由 + 负载均衡损失）
- [x] MoE 预训练（接稠密权重部分加载，已产出权重）
- [ ] 微调（SFT）/ 推理对话脚本

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
│   └── model.py            # 模型定义（MiniMindConfig / RMSNorm / Attention / MOEFeedForward / ...）
├── dataset/
│   ├── im_dataset.py       # 预训练数据集封装
│   └── packed/             # pack_corpus.py 产出的打包语料（*.i32，memmap）
├── trainer/
│   ├── train_pretrain.py   # 预训练入口
│   ├── pack_corpus.py      # 把 jsonl 流式打包成定长块（提高 token 利用率）
│   └── trainer_utils.py    # 训练工具（学习率、日志、检查点、DDP）
├── eval.py                 # 权重续写（预训练权重只认纯文本前缀，不走 chat template）
├── out/                    # 训练产出的权重
├── checkpoints/            # 续训用的完整训练状态
└── pyproject.toml
```

## 训练

在项目根目录执行。

### 1. 稠密模型预训练

```bash
python trainer/train_pretrain.py
```

数据放在 `dataset/pretrain_hq.jsonl`，每行一个 `{"text": "..."}`。
续训加 `--from_resume 1`，基于已有权重继续训练加 `--from_weight pretrain`。

用打包语料（推荐，token 利用率 76% vs 逐条 padding 的 14%）：

```powershell
.\.venv\Scripts\python.exe -u trainer\train_pretrain.py `
  --packed 1 --data_path dataset\packed\t2t_mini_packed_seq512.i32 `
  --batch_size 16 --accumulation_steps 8 --total_steps 12000 --stop_step 12000
```

### 2. MoE 预训练（接稠密权重）

MoE 把每层的 FFN 换成 **4 个专家 + 1 个 router**（top-1）。参数量涨到 3.04×，
但每个 token 仍然只过一个专家，**前向算力不变**。

```powershell
.\.venv\Scripts\python.exe -u trainer\train_pretrain.py `
  --use_moe 1 --from_weight pretrain_v3 `
  --packed 1 --data_path dataset\packed\t2t_mini_packed_seq512.i32 `
  --batch_size 16 --num_experts 4 --num_experts_per_tok 1 `
  --router_aux_loss_coef 5e-4 --save_weight pretrain_moe `
  --epochs 1 --total_steps 2000 --stop_step 2000 `
  --accumulation_steps 8 --learning_rate 5e-4 --num_workers 0
```

**`--from_weight pretrain_v3` 是「部分加载」**：`load_state_dict(strict=False)`
只继承同名同形状的参数（attention / embedding / norm，共 67 个张量），
稠密 FFN 被丢弃，MoE 的 gate + experts（104 个张量）**全新随机初始化**。
加载时会打印：

```
⚠️ 未找到 MoE 权重 pretrain_v3_512_moe.pth，回退到稠密权重 pretrain_v3_512.pth 做【部分加载】
📥 部分加载：继承 67 个张量，缺失（随机初始化）104 个，丢弃（源多出）24 个
```

**产物**（存档名会自动带 `_moe` 后缀，不会覆盖稠密权重）：

| 文件 | 说明 |
|---|---|
| `out/pretrain_moe_512_moe.pth` | 主结果（`coef=5e-4`，2000 步） |
| `checkpoints/pretrain_moe_512_moe_resume.pth` | 含优化器状态，可续训 |

**参数说明**：

| 参数 | 作用 |
|---|---|
| `--num_experts` | 每层几个专家（默认 4） |
| `--num_experts_per_tok` | 每个 token 用几个专家（top-k 的 k，默认 1） |
| `--router_aux_loss_coef` | 负载均衡损失系数（默认 5e-4）。**调大能把负载拉平**，但会让困惑度变差 |

## 查看训练结果

在项目根目录执行：

```powershell
$env:PYTHONIOENCODING="utf-8"
```

### 1. 稠密权重续写

```powershell
.\.venv\Scripts\python.exe -u eval.py --weight pretrain_v3 --use_moe 0 --preset knowledge --max_new_tokens 40
```

### 2. MoE 权重续写

```powershell
# 主结果：coef=5e-4，2000 步
.\.venv\Scripts\python.exe -u eval.py --weight pretrain_moe --use_moe 1 --preset knowledge --max_new_tokens 40

# 对照组：coef=2e-2，800 步（负载更均衡，但困惑度更差）
.\.venv\Scripts\python.exe -u eval.py --weight pretrain_moe_c2e2 --use_moe 1 --preset knowledge --max_new_tokens 40
```

**加载 MoE 权重必须加 `--use_moe 1`**，原因有两个：
① 不加的话模型按稠密构造，`load_state_dict(strict=True)` 会 size mismatch；
② 存档名的 `_moe` 后缀靠它拼出来（规则与 `train_pretrain.py` 保存时一致）。

`--weight` 填**不含 `_512` / `_moe` 的裸前缀**，脚本会自己拼成
`out/{weight}_512{_moe}.pth`。

其他可用参数：`--preset news|poem|knowledge`、`--prompt "自定义前缀"`、
`--temperature`、`--top_p`、`--max_new_tokens`、`--num_hidden_layers`。

### 3. 量化对比（困惑度 + 专家负载）

```powershell
.\.venv\Scripts\python.exe reference\moe-eval-and-load.py
```

用同一留出集（1G 语料尾部第 1,200,000 行起，400 条 / 93,670 token）对照 V3 与 MoE，
并逐层统计 4 个专家各分到多少 token。

**已产出的权重对照**：

| 权重 | 参数量 | 留出集困惑度 | 专家负载均值极差 | 生成速度 |
|---|---|---|---|---|
| `pretrain_v3`（稠密） | 30.0 M | **13.02** | — | 35~76 tok/s |
| `pretrain_moe`（coef=5e-4, 2000 步） | 91.4 M | 17.67 | 0.346（含坍缩层） | 23~46 tok/s |
| `pretrain_moe_c2e2`（coef=2e-2, 800 步） | 91.4 M | 31.06 | **0.069**（无坍缩） | — |

> 困惑度越低越好。MoE 参数 3.04× 而困惑度更差，是因为专家全新随机、只训了 2000 步；
> 而系数调大 40 倍能把负载从 0.346 压到 0.069，代价是困惑度恶化 ——
> 这是「均衡 vs 精准」的实测权衡。

## 许可证

本项目使用 [MIT License](LICENSE)。
