import os

# 必须在本模块导入 datasets/transformers 之前设置：HuggingFace 默认把缓存写到
# ~/.cache/huggingface，这里统一改到项目内 .cache/，便于清理也不占用用户目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_CACHE = os.path.join(_PROJECT_ROOT, ".cache", "huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF_CACHE, "datasets"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE, "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# datasets 的进度条在日志里刷屏，且非 tty 下会拖慢输出
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import json
import numpy as np
import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import Dataset

disable_progress_bar()


class PretrainDataset(Dataset):
    """预训练数据集：jsonl，每行至少含一个 "text" 字段。

    __getitem__ 返回 dict（而不是 tuple），这样 DataLoader 用默认 collate
    就能把整个 batch 直接拼成 {"input_ids": [B,L], "labels": [B,L], ...}，
    与 train_pretrain.py 里 batch["input_ids"] 的取法一致。
    """

    # init
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length  # 输入给GPU的最大长度

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"预训练数据不存在: {os.path.abspath(data_path)}\n"
                "请把 minimind 官方的 pretrain_hq.jsonl 放到 dataset/ 下。"
            )

        # 使用 HuggingFace datasets 的惰性加载, 避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    # __len__
    def __len__(self):
        return len(self.samples)

    # __getitem__
    def __getitem__(self, index):
        # 取出样本
        sample = self.samples[index]
        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,  # 如果长度超过max, 自动裁剪
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 一批句子里长度不同  ->  短的必须补到和最长的一样  ->  补的就是 PAD
        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        #  CrossEntropyLoss 会自动忽略 -100，不计入 loss
        # labels大致[1, 2, token_id, -100]
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # attention大致[1,1,1,0]
        # 返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()  # 非pad位置为1, pad位置为0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class PackedPretrainDataset(Dataset):
    """读取 trainer/pack_corpus.py 产出的「打包」语料（memmap）。

    与 PretrainDataset 的区别（为什么要多这一个类）：

    1. **不爆内存**：用 np.memmap 惰性映射磁盘文件，RAM 占用与语料大小无关。
       官方 PretrainDataset 走 datasets.load_dataset("json")，在 datasets 5.0.1 里
       会整文件读进 pandas —— 本机 15.7 GB RAM 上处理 1.156 GB 的
       pretrain_t2t_mini.jsonl 直接 MemoryError。

    2. **不浪费算力**：每个 block 已由多条样本打包填满（BOS…EOS BOS…EOS），
       padding 只可能出现在最后一块。对比：官方逐条补到 512 的方案，在这批平均
       205 token/条 的语料上要浪费约 60%~86% 的算力（数据越短越浪费）。

    3. **可续跑**：打包结果与随机种子无关，index -> 数据 的映射是固定的，
       所以从 checkpoint 续训时同一条数据一定落在同一个 index 上。

    labels 的 -100 掩码在这里现算，不额外占磁盘：
    打包文件里 padding 位置写的是 pad_id，读出来置为 -100 即与官方口径一致
    （CrossEntropyLoss 会忽略 -100）。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        data_path = os.path.abspath(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"打包语料不存在: {data_path}\n"
                f"请先运行: .venv\\Scripts\\python.exe trainer\\pack_corpus.py "
                f"--src <源jsonl> --seq_len {max_length}"
            )

        meta_path = data_path[: -len(".i32")] + ".meta.json" if data_path.endswith(".i32") else data_path + ".meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"缺少元信息文件: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if self.meta["seq_len"] != max_length:
            raise ValueError(
                f"打包时的 seq_len={self.meta['seq_len']} 与当前 --max_seq_len={max_length} 不一致；"
                f"请用相同的 seq_len 重新打包，或把 --max_seq_len 改成 {self.meta['seq_len']}"
            )

        # 词表必须一致，否则 embedding 行与 token id 对不上（词表错配是本项目踩过的坑）
        tok_vocab = len(tokenizer)
        if self.meta["vocab_size"] != tok_vocab:
            raise ValueError(
                f"打包时用的词表({self.meta['vocab_size']}) 与当前 tokenizer({tok_vocab}) 不一致；"
                f"请用当前 tokenizer 重新打包（否则同一 id 指向不同 token）"
            )
        self.pad_id = self.meta["pad_id"]

        n_blocks, seq_len = self.meta["n_blocks"], self.meta["seq_len"]
        self.tokens = np.memmap(data_path, dtype=np.int32, mode="r", shape=(n_blocks, seq_len))
        self.n_blocks = n_blocks

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, index):
        row = np.asarray(self.tokens[index], dtype=np.int64)
        input_ids = torch.from_numpy(row)
        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100
        attention_mask = (input_ids != self.pad_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
