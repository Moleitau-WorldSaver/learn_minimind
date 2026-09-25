import os
import sys

__package__ = "trainer"
# 让这个文件能 import 到它父目录里的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 让 warnings 把所有警告都不再打印出来（全局静音）,给训练日志去噪
warnings.filterwarnings('ignore')

#训练一个epoch,一轮数据集是一个epoch
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 记录时间
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate) # 算新的学习率
        for param_group in optimizer.param_groups: # 手动更新
            param_group['lr'] = lr

        # 把这个块里的重算子（matmul / conv / linear）自动转成低精度来算，出块时自动恢复
        with autocast_ctx: 
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # 它的精度由前向图的 dtype 自动决定，autocast 这个开关对它无效——而 scaler.scale 是反向侧的防下溢缩放，
        # 跟前向精度选择是两个正交机制，放不放进 with 块都不会改变任何精度。
        # 所以这行的位置理由是语义边界（"前向到此为止"）+ 对未来代码的防御
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            #回退原本的梯度大小
            scaler.unscale_(optimizer)
            # 裁剪上限, 防止步长过大情况出现
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            #配合优化器更新参数
            scaler.step(optimizer)
            scaler.update() # 更新下一次倍数

            optimizer.zero_grad(set_to_none=True)

        # 日志统计
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
        # 定期存档, 方便续训
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

if __name__ == "__main__":
    # 创建一个命令行参数解析器对象 
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    #声明有n多个命令, 具体实现不在此处
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # 读命令行参数，把值装进 Namespace 对象；用 args.属性名 取值（如 args.save_dir）
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # DDP 模式下：建通信组 + 把本进程绑到对应卡；并返回本进程该用的 GPU 序号
    local_rank = init_distributed_mode()
    #如果初始化了(也就是如果是DDP多卡设备), 就把args.device开始默认的0, 改成对应的local_rank
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每个进程用 args.seed+rank 作种子：8 卡各自不同（随机行为不重样），
    # 但种子完全由 args.seed 推导 → 同一条命令可复现    
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    #递归创建 args.save_dir 指向的那一串目录；如果目录已经存在，exist_ok=True 让它安静通过而不报错。
    os.makedirs(args.save_dir, exist_ok=True)
    # 读取模型参数, 只描述结构
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 是否续训(如果续训, 去读检查点, 装进ckp_data , model掌握存读)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    #优先选cuda 和 bfloat16()
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    #   造一个"精度开关"(上下文管理器对象), 供训练循环里 `with autocast_ctx:` 使用
    #   两个分支类型不同(contextlib.nullcontext vs torch.amp.autocast_mode.autocast),
    #   但都实现了 __enter__/__exit__, 所以都能用在 with 里 —— 鸭子类型, 不要求同一个类
    #     CPU -> nullcontext()        : 空壳, 什么都不做(算子在 fp32 下跑)
    #     GPU -> torch.amp.autocast() : 块内的重算子(matmul/conv/linear)自动转 dtype
    #   为什么要分支: 让循环里不写 if/else —— 一套 `with` 两种设备都能跑

    # autocast_ctx 里存的是一个"精度开关"对象：平时 5 个字段（目标 dtype、设备、是否启用、是否缓存、后端名），
    # 进 with 时临时追加 3 个旧值备份（prev / prev_fastdtype / prev_cache_enabled）供 __exit__ 恢复；
    # CPU 分支的 nullcontext 则只有一个 enter_result 字段——空壳。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb 
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        #init() 才建立与服务器的连接、创建或接上实验记录
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 用参数造出六个训练组件:
    # 大语言模型本体, 分词器, 预训练数据集对象, 分布式采样器, 梯度缩放器, 优化器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 在训练组件这里, 虽然已经开始处理数据, 但其实数据并没有处理好, 只有在取样本那一刻才真正处理好
    # 这一步代码只跑了PretrainDatase.__init__()
    # 返回的train_ds 是PretrainDataset 对象 → input_ids, labels, attention_mask
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    # 如果确认续训
    start_epoch, start_step = 0, 0
    if ckp_data:
        #读取数据
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 如果选择了使用torch.compile加速
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()