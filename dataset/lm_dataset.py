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
        # 所有记录被强制对齐成同一套键，缺的补 None
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 调用tokenizer的__call__函数, 返回一个BatchEncoding类型, 其父类是UserDict, 可以.input_ids取出token_id列表
        # 这两段的实际作用就是: 把开头标签, 和结果标签先处理
        # 主要作用是用来定位主要文本, 也就是我们只想模型学习 assistant后的内容,也就是只训练 assistant 说的内容，
        # 前面的 user 提问和角色标记全部置 -100 不参与 loss。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)


    # conversations: list[dict]，一段多轮对话 —— 每个 dict 是一条 message
    # 键集由上面的 Features schema 保证（缺的补 None）；条数不定，所以外层是 list。
    # 这段代码干了一件事 : list[dict] → 文本字符串
    def create_chat_prompt(self, conversations):
        messages = []  # 中转用的list
        tools = None
        for message in conversations: # 逐条处理
            message = dict(message)
            # 如果这条消息是 system，而且它带了工具定义
            if message.get("role") == "system" and message.get("tools"):
                # tools 可能有两种形态：JSON 文本（需解码）或已解析好的结构（直接用）。
                # 只分辨"要不要解码"，不校验内容 —— 内容合法性交给模板/下游。
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # 如果message 五个键中tool_calls不为空 且 值是str类型
            # 解码成 list -> 一个个Python对象
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message) # 将信息填入(如果信息中没有tools和tool_call 就将原本的填入)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids) # 先全部掩掉
        i = 0
        while i < len(input_ids): #从前往后遍历
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start # 寻找结尾
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1 # 结尾找到, 由于左闭右开, end要+1
                #比较end+len(eos_id) 和max_length哪个小, 防止越界
                for j in range(start, min(end + len(self.eos_id), self.max_length)): 
                    labels[j] = input_ids[j]
                #为找下一段做准备
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1 # i++遍历
        return labels

    def __getitem__(self, index):
        # 结构（三层，全由上面 features 规定）：
        #   sample                   = {'conversations': [...]}        ← 1 个键
        #   sample['conversations']  = list，长度不定（实测 2~16）      ← 元素个数 = 对话条数
        #   ...[i]                   = message dict，键集固定 5 个：    ← 键集由 features 内层规定
        #                              role / content / reasoning_content / tools / tool_calls
        # 也就是说我们想取得信息得 samples[i]['conversations'][1]['content']
        #                                      ↑字符串键     ↑整数    ↑字符串键
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        #  list[dict] → 文本字符串
        prompt = self.create_chat_prompt(conversations) # 渲染
        # 空思维链是因为渲染模板固定的
        # assistant 消息没有 reasoning_content → 渲染成空块
        # 移除空思维链
        prompt = post_processing_chat(prompt)
        # 至此 prompt 是samples[index] 这条样本（一整段多轮对话）渲染后的文本,类型是 str
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] #截断超过的
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) #用padding补全缺少的
        # 这里的input_ids 把用户的输入一并拿到了
        labels = self.generate_labels(input_ids)
        # 返回两个张量的input_ids, lables
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

