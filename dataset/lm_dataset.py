from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # 工具对话: 不注入人设 system 提示。
    # 理由: SYSTEM_PROMPTS 全是人设措辞, 而工具对话的 system 槽位承载的是
    #       工具 schema(模板渲染成 <tools>...</tools>)。往里插人设会污染 schema,
    #       并给模型发出矛盾信号(泛泛的人格描述 vs 严格的调用格式命令)。
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 20%概率性添加system(真实概率可能为18%左右)
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签(如果用户表态,就一定删除)
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            # 移除空思考
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        # 返回的数据集对象HuggingFace, samples是其 实例
        # 它像 list 一样支持 len() 和 [i]，但返回的是 dict，还多了 column_names / features 这些"列"的概念——它是数据源
        # samples[i]["text"]拿取的是每一行的文本
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        # 分词 由文本到 token_id
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False, #不加特殊符号
            max_length=self.max_length - 2,  # 上限就是最长长度 - 2, 预留 BOS + EOS 的位置
            truncation=True, # 超长就截断
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        # input_ids 是list类型, 后续用padding补全 ,确保batch相等
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        #转化为张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        #         CrossEntropyLoss 会自动忽略 -100，不计入 loss
        # 克隆下正确的信息
        labels = input_ids.clone()
        #所有PAD的位置变成-100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids