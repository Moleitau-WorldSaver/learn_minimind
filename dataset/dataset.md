# 数据集台账

记录本项目用过的每一份数据，避免分不清某个 `out/*.pth` 是在什么数据上训出来的。

## 当前文件

| 文件 | 来源 | 条数 | 体积 | 状态 |
|---|---|---|---|---|
| `pretrain_hq.jsonl` | 旧命名体系，疑似官方早期 `pretrain_hq` 的前 6000 行切片 | 6,000 | 1.58 MB | ⚠️ 太小，只能冒烟测试 |
| `pretrain_t2t_mini.jsonl` | `jingyaogong/minimind_dataset` 官方主线 mini 版（2026-09-20 下载完成） | **1,270,238** | 1183.55 MB | ✅ 全量数据，本机跑不完一个 epoch |
| `pretrain_10k.jsonl` | 上面文件的前 10000 行切片 | 10,000 | 7.11 MB | ✅ 快速验证 |
| `pretrain_30k.jsonl` | 上面文件的前 30000 行切片 | 30,000 | 21.37 MB | ✅ 已训完 1 epoch（见下） |
| `pretrain_100k.jsonl` | 上面文件的前 100000 行切片 | 100,000 | 70.22 MB | ✅ 长训用 |

### 已完成的训练

| 权重 | 数据 | 步数 | loss | 说明 |
|---|---|---|---|---|
| `out/pretrain_512.pth` | `pretrain_hq.jsonl` | 188 | 5.96 → 3.45 | ⚠️ 1.58MB 数据上的**过拟合**，loss 低但不代表能力强 |
| `out/pretrain_v2_512.pth` | `pretrain_30k.jsonl` | **938** | **8.17 → 5.92** | ✅ 首次真实训练，lr 完整退火 5e-4 → 5e-5 |
| `out/pretrain_v3_512.pth` | `pretrain_t2t_mini.jsonl`（1G，打包） | **12,000** | **7.18 → 2.57** | ✅✅ 首个真正可用的模型，见 `reference/training-report-v3.md` |

**⚠️ loss 数值不可跨方案比较**：v1/v2 的 loss 是「逐条 padding 到 512 + `-100` 掩码」口径
（这批语料平均只有 70~228 token/条，**86% 的位置是 padding**）；v3 用**打包**口径
（利用率 76.3%，几乎没有 padding）。两者算的不是同一件事，比 loss 必须用同一口径的
留出集困惑度 —— 见下表。

### 留出集评测（同一口径，可比）

留出集 = `pretrain_t2t_mini.jsonl` 第 1,200,000~1,200,399 行（400 条 / 93,670 个 token），
已用 `reference/row-to-block-check.py` 证明落在打包第 680,942~681,196 块，
远超训练消耗的 192,000 块 —— **模型没训过**。复现：`reference/heldout-ppl-compare.py`

| 模型 | 交叉熵 | 困惑度 | 相对随机 |
|---|---|---|---|
| 随机初始化 | 8.8569 | 7022.39 | 100.0% |
| v1（hq 6000 条） | 8.2262 | 3737.74 | 92.9% |
| v2（30k 条） | 5.9466 | 382.47 | 67.1% |
| **v3（1G 打包，12000 步）** | **2.5666** | **13.02** | **29.0%** |

### 打包语料（`trainer/pack_corpus.py` 产出）

`datasets.load_dataset("json", ...)` 在 datasets 5.0.1 里走 pandas 整文件读入，
处理 1.156 GB 的 `pretrain_t2t_mini.jsonl` 会 **MemoryError**（本机 RAM 15.7 GB、可用仅 5.1 GB）。
所以改用流式打包，产物在 `dataset/packed/`：

| 文件 | 内容 |
|---|---|
| `t2t_mini_packed_seq512.i32` | 1.517 GB，740,731 块 × 512 = 379,254,272 格 |
| `t2t_mini_packed_seq512.meta.json` | 元信息（源文件指纹 / 行数 / token 数 / 词表 / 利用率），`PackedPretrainDataset` 会校验 |

源文件 1,270,237 行 / 289,396,513 token / 平均 227.8 token/条；
打包后 **token 利用率 76.31%**（官方逐条 padding 方案在同一批语料上只有 13.9%）。


### 切片方法（务必用二进制读写，不要用 `Set-Content -Encoding UTF8`）

PowerShell 5.1 的 `Set-Content -Encoding UTF8` **会写入 BOM**（`EF BB BF`），导致 `datasets` 报
`ArrowInvalid: JSON parse error: Invalid encoding in string. in row 0`。纯文本检查（`StartsWith('{"text":')`）**发现不了**这个问题。

正确做法（二进制逐行搬运，零编码转换）：

```python
with open(SRC, "rb") as fin, open(DST, "wb") as fout:
    for i, raw in enumerate(fin):
        fout.write(raw)
        if i + 1 >= N: break
```

验证切片是否干净（首字节必须是 `123` = `{`）：

```powershell
Get-Content dst.jsonl -Encoding Byte -TotalCount 4   # 应为 123 34 116 101
```

### 另一个 Windows 编码坑

本机 `open(path, encoding="utf-8")` 的**默认编码是 GBK**（`PYTHONIOENCODING` 只影响 stdout/stderr，不影响 `open()`）。
读 UTF-8 数据文件时必须显式写 `encoding="utf-8"`，或用二进制模式，否则会在第一个汉字处抛
`UnicodeDecodeError: ... invalid start byte`。**这个报错不代表文件坏了。**

## 格式要求

每行一个 JSON 对象，**字段名必须是 `text`**：

```json
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
```

硬性约束来自两处代码：

- `dataset/im_dataset.py:55` —— `str(sample["text"])`，字段名写死为 `text`
- `dataset/im_dataset.py:43` —— `load_dataset("json", data_files=data_path, split="train")`

如果从别处下的数据字段名是 `content` / `raw_content` 之类，必须先转换，否则会在 `sample["text"]` 抛 `KeyError`。

## 官方数据集全貌（`jingyaogong/minimind_dataset`）

Apache-2.0 + CC-BY-NC-2.0，HF 上 5662 下载 / 113 赞。

| 文件 | 大小 | 阶段 |
|---|---|---|
| `pretrain_t2t.jsonl` | 7891.73 MB | 完整预训练 |
| `pretrain_t2t_mini.jsonl` | 1183.55 MB | 快速复现预训练（本项目采用） |
| `sft_t2t.jsonl` | 13443.01 MB | 完整 SFT |
| `sft_t2t_mini.jsonl` | 1658.63 MB | 快速 SFT |
| `dpo.jsonl` | 51.17 MB | 偏好对齐 |
| `agent_rl.jsonl` | 78.24 MB | Agent / 工具调用 |
| `rlaif.jsonl` | 22.65 MB | RLAIF |
| `lora_identity.jsonl` | 0.02 MB | LoRA 示例（身份） |
| `lora_medical.jsonl` | 32.43 MB | LoRA 示例（医疗） |
| `lora_exam.jsonl` | 23.51 MB | LoRA 示例（考试） |

官方说明的数据来源：通用文本语料 + 对话整理语料 + 蒸馏补充语料，经清洗、去重、长度控制、格式统一后进入训练。上游为 [匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)、[Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align) 等宽松协议数据源（主要在 ModelScope）。

## 本项目已知的数据相关问题

1. **词表不一致（历史遗留）**：`logs/pretrain_demo_run.log:9` 记录那次训练用的 tokenizer 是 **3272** 词表，而当前 `model/tokenizer.json` 是 **6400** 词表（与 `MiniMindConfig` 默认 `vocab_size=6400` 一致）。
   - 后果：demo 那次训练出的语言建立在 **3272 词表的坐标系**上，与现在的 6400 词表**同一 id 指向不同 token**，不能直接续训。
   - **【2026-09-21 更正】早期本文件写过「`out/pretrain_512.pth` 后 3128 行从未被训练过」——这条是错的。**
     实测（`reference/embed-rows-check.py`）：前 3272 行的平均参数变化 0.004224，
     后 3128 行 0.004397，比值 0.96，**两段都被训练过**。
     原因：`train_pretrain.py` 只在 `tok_vocab > lm_config.vocab_size` 时才改模型，
     3272 < 6400 所以 embedding 保持 6400 行，forward 时 6400 行**全部**参与计算、全部拿到梯度。
     错在把「tokenizer 只有 3272 个 id」误读成「另外 3128 行不参与训练」——
     没人用的行照样会被算进 forward 与 backward，只是它们对应的 token 永远不会出现在输入里。

2. **旧数据量级不足**：`pretrain_hq.jsonl` 只有 831,784 字符，约等于 2~3 本书。6000 条样本上 loss 降到 3.45 是**过拟合**，不是学会了语言。
   （留出集困惑度实测：v1 = 8.23，几乎等于随机初始化的 8.86 —— 确实没学会。）

3. **下载来源**：`https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/<文件名>`

4. **1G 文件不能用 `PretrainDataset` 直接读**：`datasets` 5.0.1 的 json builder 走 pandas
   整文件读入，1.156 GB 的 `pretrain_t2t_mini.jsonl` 在本机（RAM 15.7 GB / 可用 5.1 GB）
   必抛 `MemoryError`。改用 `dataset/packed/` 下的打包文件 + `--packed 1`。

## 训练命令备忘

```powershell
# 冒烟测试（60 步，约 2 分钟）
.\.venv\Scripts\python.exe trainer\train_pretrain.py --save_weight pretrain_mini --stop_step 60 --log_interval 20

# 正式训练（注意 --save_weight 必须改，否则覆盖 out/pretrain_512.pth）
.\.venv\Scripts\python.exe trainer\train_pretrain.py --data_path dataset\pretrain_30k.jsonl --save_weight pretrain_v4
```

`--max_seq_len` 官方 pretrain 用 340（注释说明「中文 1 token ≈ 1.5~1.7 字符」）。
**注意**：上面第 2 条里 `--data_path dataset\pretrain_t2t_mini.jsonl`（1G 文件）在旧代码里
会 MemoryError，见下面新流程。

### 【2026-09-21 起推荐】打包语料 + 固定步数训练

1G 文件走两步：

```powershell
# 第 1 步：流式打包（一次性，约 7 分钟；只读源文件，产物写到 dataset\packed\）
#   1G 全量：1,270,237 行 -> 740,731 块 x 512，token 利用率 76.31%
.\.venv\Scripts\python.exe trainer\pack_corpus.py --src dataset/pretrain_t2t_mini.jsonl --seq_len 512 --name t2t_mini_packed

# 第 2 步：训练（实测 3.3 step/s，12000 步约 56 分钟）
.\.venv\Scripts\python.exe -u trainer\train_pretrain.py `
  --packed 1 `
  --data_path dataset\packed\t2t_mini_packed_seq512.i32 `
  --save_weight pretrain_v4 `
  --batch_size 16 `
  --accumulation_steps 8 `
  --max_seq_len 512 `
  --epochs 1 `
  --total_steps 12000 `
  --stop_step 12000 `
  --log_interval 200 `
  --save_interval 4000
```

关键点（每条都对应一个踩过的坑）：

- **`--total_steps` 必须给**：lr 按「完成比例」余弦退火。1G 数据一个 epoch = 46296 step，
  若退火总长按这个算，训到 12000 步时 lr 才降到约 60%，**等于没退火完**，loss 末尾降不下去。
  给了 `--total_steps 12000` 后 lr 精确从 5e-4 退到 5e-5。
- **`--batch_size 16` 不是 32**：实测 8GB 显存上 batch=16 反而**最快**（0.298 s/step）且只占 4.7GB；
  batch=32 涨到 1.688 s/step 还贴边 7.4GB（`reference/vram-scan.py`）。
- **续训**：加 `--from_resume 1` 从 `checkpoints/<save_weight>_<hidden>_resume.pth` 恢复
  模型 + 优化器 + 进度。它与 `--from_weight` 是两套东西：前者读 `checkpoints/`，后者读 `out/`。
- **`--save_interval 4000`**：每 4000 步落盘；训练中途 Ctrl+C 也会自动保存进度。

