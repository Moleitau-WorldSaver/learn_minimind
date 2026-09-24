import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model import MiniMindForCausalLM


# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 日志
def Logger(content):
    if is_main_process():
        print(content)


# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    return (
        lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    )  # step=0 时 lr=lr，step=end 时降到 0.1*lr


# 初始化分布式
def init_distributed_mode():
    #从环境变量取一个"RANK" 的值；如果这个变量没设, 就返回 -1
    #这里的RNAK只是用来当标志位的
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    #如果设置了 RANK, 进行下面操作
    #初始化 PyTorch 分布式通信（默认进程组），让多个进程能互相同步数据；backend="nccl" 指定用 NVIDIA GPU 专用的通信后端。
    dist.init_process_group(backend="nccl")
    # 从环境变量里取出本机进程的 GPU 序号（字符串），转成整数存进 local_rank
    #注意: LOCAL_RANK 和 上面的 RANK 不是一个环境变量(单机上两者相同)
    # LOCAL_RANK 是本机第几块卡
    # RANK是进程在进程组里的序号
    local_rank = int(os.environ["LOCAL_RANK"])
    #训练开始, torch会创建子进程, 每个子进程的LOCAL_RANK不一样, 开始绑卡
    torch.cuda.set_device(local_rank)
    return local_rank


# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    # np种子, 返回数组
    np.random.seed(seed)

    #设备种子(其实真正有效的是第一行)
    torch.manual_seed(seed) #CPU + 所有 GPU + MPS + XPU
    torch.cuda.manual_seed(seed) #只有当前那一块 GPU
    torch.cuda.manual_seed_all(seed) #所有 GPU

    torch.backends.cudnn.deterministic = True #只用确定性算法 (开启), 默认是False
    torch.backends.cudnn.benchmark = False #自动挑选最快算法 (关闭), 默认就是False
    # 算法选定 + 算法内部确定,确保可复现(同一份输入 -> 同一份输出)
    # 开始确定算法是为了规避浮点先后相加不一致的问题
    # 戏剧性的例子:
    # a, b, c = 1e16, -1e16, 1.0
    # (a + b) + c  =  1.0      # 先抵消成 0，再 +1 → 1.0
    # a + (b + c)  =  0.0      # 先算 b+c，1.0 被 -1e16 吃掉 → 再抵消，剩 0
    # 确定性算法开启 要堵的是所有「并行顺序影响累加顺序」的路径
    #总结: 确定性换的是「可复现」，不是精度，代价是几个百分点的速度;
    # benchmark 换的是「速度」，代价是启动开销和可复现性
    
# 设置检查点
def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="../checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    # 如果有且开启了moe, 检查点文件名要加上moe后缀,造一个,如果没开启就空字符串
    moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    #拼成完整的检查点名字
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth" # 纯权重文件
    # 权重(fp32) + 训练状态；体积≈5.6倍，两块来源:
    #   ① fp32 权重 127MB  (fp16 的 2 倍, 续训必须全精度)
    #   ② AdamW 动量 229MB (exp_avg + exp_avg_sq, 每个和模型一样大) ← 最大头
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth" 

    #todo ========== AdamW 优化器 ==============
    # 干什么: 拿到梯度后决定"怎么更新参数"。每一步的实际步长 = lr × 自适应系数
    #   lr 是局一个值(你给的), 后面用 optimizer.param_groups['lr'] 手动调度
    #   自适应系数  每个参数独立, 由历史梯度算出:
    #    所以, 实际步长 = lr × 系数, 系数量级为 1 → 步长由 lr 定标, 不随梯度大小漂移
    #    → 第 1 步严格 = 1; 之后随梯度方向变乱而变小(方向不明就自动走小步)
    #
    # 名字里的 W = 解耦的权重衰减 (decoupled weight decay):
    #   老 Adam 把 weight_decay 混进梯度里 → 会被 1/√v 一起缩放, 衰减力度被扭曲
    #   (梯度大的参数衰减变弱、梯度小的变强 —— 不是你要的)
    #   AdamW 把它拆出来直接作用在参数上 → 力度恒定, 泛化更好, 成为 Transformer 标配
    #
    # 代价: 每个参数要养 exp_avg + exp_avg_sq 两个动量张量, 各自和参数一样大
    #       → 这就是 _resume.pth 比 .pth 大 5 倍多的主因
    #
    # 两级结构对应 optimizer.state_dict() 的两半:
    #   param_groups → 全局(lr/betas/eps/weight_decay)   你给的
    #   state        → 个体(step/exp_avg/exp_avg_sq)     算法攒的
    #todo ========== AdamW 优化器 ===================


    # 判断 model 是不是被 DDP 包过——是的话要先扒掉那层壳取里面的真模型，
    # 因为 DDP 的 state_dict() 会给所有参数名加上 module. 前缀。
    if model is not None:
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        #先把权重写入临时文件, 写完正式改名, 保障写入时崩溃仍旧保存是完整的旧权重,而不是半个
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 如果上传日志到wandb(这里用的swanlab)
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"): # 这个库有没有get_run接口
                run = wandb.get_run() #得到当前实验的句柄
                # 根据句柄取出id, 如果没找到默认None
                wandb_id = getattr(run, "id", None) if run else None
            else: #如果没有get_run接口, 直接拿id
                wandb_id = getattr(wandb, "id", None)
        # 断言 optimizer 不是 None
        # optimizer 是一个 AdamW 对象——你把自己模型的全部参数交给它，之后由它负责「拿着梯度把参数更新掉」
        assert optimizer is not None
        # 用字典打包好恢复训练所需的一切
        resume_data = {
            "model": state_dict, #模型的参数字典
            "optimizer": optimizer.state_dict(), #优化器攒下的全部记忆
            "epoch": epoch, # 训练到第几轮 → 恢复后作为 range 的新起点
            "step": step, # 【本轮内】已走完多少步 → 恢复后跳过前 N 个 batch
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                            # 当时用了几张卡 → 换卡数续训时按比例换算 step
            "wandb_id": wandb_id, # 实验记录的身份证 → 续训时接回同一条日志曲线
        }

        #kwargs 装的是「调用时传了、但函数签名里没有显式列出的」那些关键字参数——在你项目里，实际只有 scaler 一个
        # scaler 是 torch.cuda.amp.GradScaler 的实例——混合精度训练用的「梯度缩放器」,用它做loss的放大以防止fp16梯度下溢
        # 用key, value 来遍历 kwargs.items() 是「字典 kwargs 的 items」，GradScaler 是装在里面的那个值
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):# 有 state_dict + 是 DDP, 扒壳后的状态字典
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict() # 有 state_dict + 不是 DDP, 状态字典
                else:
                    resume_data[key] = value #没有 state_dict, 原样写入

        #同样的续训也是先传入临时，再改名
        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:  # 加载模式(model 其实是控制开关, 开就是存档, 关就是读档)
        #先检查存不存在
        if os.path.exists(resume_path):
            #把 _resume.pth 从磁盘反序列化读回来（得到一个 dict），并强制把所有张量先放到 CPU 内存
            # 如果CPU不够? 直接崩了
            ckp_data = torch.load(resume_path, map_location="cpu")
            #world_size 存的是「保存这个 checkpoint 那一刻，一共有几个训练进程（几张卡）」, 默认是1
            saved_ws = ckp_data.get("world_size", 1)
            #读出「现在这次运行有几个训练进程（几张卡）」；如果进程组没建（单卡跑），就当作 1
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws: #如果不相等,按照current_ws开始跑, 并打日志
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )

            return ckp_data
        return None


# 初始化模型
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path="../model",
    save_dir="../out",
    device="cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

        # 【改动点】接「稠密权重 → MoE」时，带 _moe 后缀的文件并不存在
        #（稠密权重叫 pretrain_v3_512.pth，MoE 才叫 ..._512_moe.pth）。
        # 而 load_state_dict(strict=False) 本来就能做部分加载：
        # 只有 attention / embedding / norm 这些同名同形状的参数会被继承，
        # 稠密 FFN 被丢弃、MoE 的 gate+experts 保持随机初始化 —— 这正是官方做法。
        # 所以这里加一次回退：带后缀的找不到，就试不带后缀的。
        if not os.path.exists(weight_path) and moe_suffix:
            fallback = f"{save_dir}/{from_weight}_{lm_config.hidden_size}.pth"
            if os.path.exists(fallback):
                Logger(
                    f"⚠️ 未找到 MoE 权重 {os.path.basename(weight_path)}，"
                    f"回退到稠密权重 {os.path.basename(fallback)} 做【部分加载】："
                    f"attention/embedding/norm 继承，MoE 专家随机初始化"
                )
                weight_path = fallback

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"权重文件不存在：{weight_path}\n"
                f"请先训练出该权重（保存目录：{save_dir}），"
                f"或把 --from_weight 改为 none 从零开始训练。"
            )

        weights = torch.load(weight_path, map_location=device)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing or unexpected:
            Logger(
                f"📥 部分加载：继承 {len(weights) - len(unexpected)} 个张量，"
                f"缺失（随机初始化）{len(missing)} 个，丢弃（源多出）{len(unexpected)} 个"
            )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    # 通过 Module.to 的显式调用避免类型检查器将 `device` 误判为 self 参数。
    return torch.nn.Module.to(model, device=torch.device(device)), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler  #
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []  # 当前批次
        skipped = 0  # 已跳过的批次数

        for idx in self.sampler:
            batch.append(idx)  # 添加样本到当前批次

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1  # 增加跳过计数
                    batch = []  # 清空批次，不返回
                    continue  # 跳过这个批次

                yield batch
                batch = []  # 重置批次

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.skip_batches)