import torch
from torch import optim, nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 初始化, 确保w + BA = 原来的 W 因此必须有一个矩阵全为 0 

        # 矩阵A高斯初始化
        # normal_ 是把矩阵所有的数随机, mean是随机的数的平均值(0), std是标准差, 是正态分布的散布程度,mean ± (x)std 
        # mean 和 std 这两个都是写死的常数
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    # 遍历所有子模块
    for name, module in model.named_modules():
        # 只挑方阵线性层 —— 在当前结构下等价于"只给 attention 的 q_proj / o_proj 加 LoRA"
        # （k/v 因 GQA 变 768→384，FFN 是 768→2432，lm_head 是 768→6400，都不是方阵）
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # lora装着是AB小矩阵的一个对象
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # 把lora 注册成为module 子块
            setattr(module, "lora", lora)
            # 备份原始的 forward：下一行 module.forward = forward_with_lora 会覆盖它，
            # 覆盖后就再也拿不到原实现 —— 而新 forward 要靠它算 Wx 那一半   
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(input, layer1=original_forward, layer2=lora):
                return layer1(input) + layer2(input)

            module.forward = forward_with_lora

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)