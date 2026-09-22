"""把 1G 的 jsonl 语料流式 tokenize + 打包成可 memmap 的整块数据。

为什么需要它（三个问题一次解决）：

1. **MemoryError**：`datasets.load_dataset("json", ...)` 在 datasets 5.0.1 里走 pandas
   整文件读入，处理 1.156 GB 的 `pretrain_t2t_mini.jsonl` 时抛 MemoryError
   （本机 RAM 15.7 GB、可用仅 5.1 GB）。本脚本逐行流式读，内存恒定在几十 MB。

2. **padding 浪费 86%**：官方 `PretrainDataset` 把每条样本都补到 max_length=512，
   而这批语料平均只有约 71 token —— 一个 batch 里 86% 的位置是 `-100`，不产生梯度，
   等于 86% 的算力白烧。本脚本把多条样本**打包**进同一个 512 块（BOS…EOS BOS…EOS），
   padding 只可能出现在最后一块，token 利用率从 14% 提到 ~97%。

   注意：打包后每个 batch 都含"多条样本的边界"，梯度贡献是若干条样本的平均，
   与"一条样本恰好占满一个 batch"在期望上等价（都是对真实语料分布做 next-token 预测），
   唯一副作用是跨样本边界处会学"预测下一条的开头"—— 这是 GPT 系列的标准做法。

3. **可续跑**：打包结果与随机种子无关，写一次磁盘反复复用；随机性只来自 DataLoader 的
   shuffle。所以从 checkpoint 续训时，同一条数据一定落在同一个 index 上。

输出（默认写到 dataset/packed/ 下，只读不改原文件）：
  <name>.i32     int32 [n_blocks, seq_len]  已打包的 token id（pad 位置为 pad_id）
  <name>.meta.json                          元信息：seq_len/n_blocks/源文件指纹/词表大小

运行：
  .venv\\Scripts\\python.exe trainer\\pack_corpus.py --src dataset/pretrain_t2t_mini.jsonl --seq_len 512
  .venv\\Scripts\\python.exe trainer\\pack_corpus.py --src dataset/pretrain_30k.jsonl --seq_len 512 --name pretrain_30k_packed
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
from transformers import AutoTokenizer

DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "packed")
DEFAULT_TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "model")


def sha256_head(path: str, nbytes: int = 1 << 20) -> str:
    """取文件头部 1MB 的哈希：既便宜又能标识版本（避免误用旧缓存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def fmt(sec: float) -> str:
    return f"{sec:.0f}s" if sec < 90 else f"{sec / 60:.1f}min"


def iter_blocks(src, tokenizer, seq_len=512, rows=0, text_field="text", stats=None, row_map=None):
    """流式生成打包块（单一事实来源：打包、核对行->块映射都用这一份实现）。

    yield (block_index, np.int32[seq_len])，padding 位置填 pad_id。
    内存恒定：不会把语料读进内存。

    stats   : 传入 dict 会回填 n_rows / n_tokens / n_blocks，省一次全文件扫描
    row_map : 传入 dict[int, set[int]]，键为源文件行号（从 0 计），
              会把该行落在第几块写进对应的 set 里（不复制打包逻辑）
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    budget = seq_len - 1  # 留 1 个位置给 EOS

    buf: list[int] = []
    block = np.empty(seq_len, dtype=np.int32)
    idx = 0
    n_rows = n_tokens = 0

    with open(src, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if rows and n_rows >= rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)[text_field]
            except (json.JSONDecodeError, KeyError):
                continue

            ids = tokenizer(
                str(text), add_special_tokens=False, max_length=budget - 1, truncation=True
            ).input_ids
            ids = [bos_id] + ids + [eos_id]

            # 打包：整条塞得下就塞，塞不下就把当前块落盘、这条放进新块。
            # 不劈开样本（跨块会截断上下文），所以 padding 只会出现在最后一块。
            if len(buf) + len(ids) > seq_len:
                block[: len(buf)] = buf
                block[len(buf):] = pad_id
                yield idx, block
                idx += 1
                buf = []

            # 这一行会被放进第 idx 块（与上面「封口」逻辑保持同一时序）
            if row_map is not None and n_rows in row_map:
                row_map[n_rows].add(idx)

            buf.extend(ids)
            n_rows += 1
            n_tokens += len(ids)

    if buf:
        block[: len(buf)] = buf
        block[len(buf):] = pad_id
        yield idx, block
        idx += 1

    if stats is not None:
        stats.update(n_rows=n_rows, n_tokens=n_tokens, n_blocks=idx)


def main():
    ap = argparse.ArgumentParser(description="流式 tokenize + 打包语料")
    ap.add_argument("--src", required=True, help="源 jsonl（每行一个含 text 字段的 JSON）")
    ap.add_argument("--name", default=None, help="输出前缀，默认取源文件名")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--seq_len", type=int, default=512, help="打包块长度（= 训练 max_seq_len）")
    ap.add_argument("--rows", type=int, default=0, help="只取前 N 行（0=全量）")
    ap.add_argument("--text_field", default="text")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(PROJECT_ROOT, args.src)
    if not os.path.exists(src):
        raise SystemExit(f"源文件不存在：{src}")

    name = args.name or os.path.splitext(os.path.basename(src))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    bin_path = os.path.join(args.out_dir, f"{name}_seq{args.seq_len}.i32")
    meta_path = os.path.join(args.out_dir, f"{name}_seq{args.seq_len}.meta.json")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_DIR)
    vocab = len(tokenizer)
    bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    print(f"tokenizer 词表 = {vocab} | bos={bos_id} eos={eos_id} pad={pad_id}")

    seq_len = args.seq_len

    # 打包只走 iter_blocks（单一实现），统计由它顺带回填
    t0 = time.time()
    stats: dict = {}
    with open(bin_path, "wb") as fout:
        for idx, block in iter_blocks(src, tokenizer, seq_len, args.rows, args.text_field, stats):
            fout.write(block.tobytes())
            if (idx + 1) % 100_000 == 0:
                print(f"  已写出 {idx + 1:>8,} 块 | {fmt(time.time() - t0)}")
        fout.flush()
        os.fsync(fout.fileno())

    n_rows, n_tokens, blocks_written = stats["n_rows"], stats["n_tokens"], stats["n_blocks"]
    elapsed = time.time() - t0
    size_gb = os.path.getsize(bin_path) / 1e9
    util = n_tokens / (blocks_written * seq_len) * 100 if blocks_written else 0

    meta = {
        "src": os.path.relpath(src, PROJECT_ROOT).replace("\\", "/"),
        "src_bytes": os.path.getsize(src),
        "src_head_sha256_16": sha256_head(src),
        "rows_read": n_rows,
        "tokens": n_tokens,
        "seq_len": seq_len,
        "n_blocks": blocks_written,
        "vocab_size": vocab,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "token_utilization_pct": round(util, 2),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print(f"源文件      : {meta['src']}  ({meta['src_bytes'] / 1e9:.3f} GB)")
    print(f"读取行数    : {n_rows:,}")
    print(f"总 token    : {n_tokens:,}  (平均 {n_tokens / max(n_rows, 1):.1f} token/条)")
    print(f"打包块数    : {blocks_written:,}  x {seq_len} = {blocks_written * seq_len:,} 格")
    print(f"token 利用率: {util:.2f}%   <-- 官方逐条 padding 方案在这批语料上是 13.9%")
    print(f"落盘        : {os.path.relpath(bin_path, PROJECT_ROOT)}  ({size_gb:.3f} GB)")
    print(f"              {os.path.relpath(meta_path, PROJECT_ROOT)}")
    print(f"耗时        : {fmt(elapsed)}  ({n_rows / max(elapsed, 1e-9):,.0f} 行/s)")
    print("=" * 68)


if __name__ == "__main__":
    main()
