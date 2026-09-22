import os
import sys

# 训练日志里有中文和 emoji，Windows 控制台默认 GBK 会直接抛 UnicodeEncodeError 打断训练，
# 所以先把标准输出切到 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import argparse
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# 注意：im_dataset 必须在 datasets 之前导入，它负责把 HuggingFace 缓存指到项目内 .cache/
from dataset.im_dataset import PackedPretrainDataset, PretrainDataset
from model.model import MiniMindConfig
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")

# 默认路径以项目根目录为基准，这样在根目录或 trainer/ 目录下启动都指向同一处
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "pretrain_hq.jsonl")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "out")
DEFAULT_TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "model")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# 记录「最近一次真正完成参数更新的 step」，供 Ctrl+C 时保存进度使用。
# 用 list 而不是全局变量，是为了在 train_epoch 里能直接改到（不需要 global 声明）。
_LAST_STEP = [0]


def save_state(epoch, step, reason=""):
    """把「模型 + 优化器 + 进度」落盘，用于续训。

    这里统一走一个函数，是为了修掉两个真实存在的续训缺陷（2026-09-21 实测）：
      ① 原先只在 `step % save_interval == 0` 时保存，训练中途被打断（Ctrl+C / 断电 /
         笔记本休眠）时最后一段进度全丢；
      ② 训练结束时写进 checkpoint 的 `epoch` 是「刚跑完的那个 epoch」，
         续训时 `epoch == start_epoch and start_step > 0` 命中，SkipBatchSampler 会
         把这一整轮数据全部跳掉 —— 一步都不训，还以退出码 0 正常退出。
         现在写完一轮记的是 `epoch + 1`，语义变成「下一个该跑的 epoch」。
    """
    model.eval()
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir=CHECKPOINT_DIR,
    )
    model.train()
    if reason:
        Logger(f"  ↳ 已保存训练状态（{reason}）：epoch={epoch} step={step}")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()

    # 遍历批次循环
    for step, batch in enumerate(loader, start=start_step + 1):

        if args.stop_step and step > args.stop_step:
            Logger(f"--stop_step={args.stop_step}，提前结束本轮（仅用于冒烟测试）")
            # 关键：提前结束时也要把进度落盘，否则这段训练白跑
            save_state(epoch, _LAST_STEP[0], f"stop_step={args.stop_step}")
            return

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # 将数据移动到指定设备，一般是GPU
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)

        # 计算当前学习率
        # lr 的退火总长用 args.total_steps 显式给定（0 = 退回「epochs * iters」的老口径）。
        # 为什么需要它：学习率是按「完成比例」余弦退火的，如果退火总长与实际要跑的步数
        # 不一致（例如打算只跑 12000 step，却按 1 个 epoch=39695 step 退火），
        # 训练结束时 lr 只降到约 60%，等于没退火完 —— 这正是 2026-09-20 那次
        # pretrain_v2 训练 loss 停在 5.92 降不下去的原因之一。
        lr_total_steps = args.total_steps if args.total_steps > 0 else args.epochs * iters
        lr = get_lr(epoch * iters + step, lr_total_steps, args.learning_rate)

        # 把动态学习率放进优化器
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr  # 更新优化器学习率

        with autocast_ctx:  # 混合精度上下文
            # 前向传播
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # 计算loss（稠密模型时 aux_loss 恒为 0）
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps  # 平均化损失，适应梯度累计

        scaler.scale(loss).backward() #这里累加

        # 特定次数后,开始处理, 更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # scaler.unscale_(): 还原梯度的真实值
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            # scaler.step(): 执行参数更新
            # scaler.update(): 更新scaler的缩放因子
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 记下「已完成参数更新的最新 step」：只有跨过累积边界，梯度才真的写进了参数，
        # Ctrl+C 时据此保存进度才不会把没生效的梯度算进去。
        if step % args.accumulation_steps == 0:
            _LAST_STEP[0] = step

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
            current_lr = optimizer.param_groups[-1]["lr"]  # 当前学习率

            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            # 记录到实验跟踪系统
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到评估模式

            # 构建保存路径
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            # DDP模型需要通过.module访问真正的模型
            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度保存，减少存储空间
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)

            # 保存完整训练状态（含优化器、进度），用于续训
            save_state(epoch, step)
            model.train()  # 恢复训练模式


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, help="模型保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="数据加载线程数；Windows 上多进程 worker 容易卡死，故默认 0",
    )

    # ========== 训练策略参数 ==========
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="训练的最大截断长度")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）；1 时把每层的 FFN 换成 4 专家 + 路由（实测参数量约 3.0 倍、每 token 算力不变）",
    )

    # ========== 数据和恢复参数 ==========
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="预训练数据路径")
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")

    # ========== 数据格式与学习率退火 ==========
    parser.add_argument(
        "--packed",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=使用 trainer/pack_corpus.py 产出的打包语料（*.i32，memmap，不占内存、不浪费 padding）；"
        "0=使用原来的逐条 padding jsonl 流程",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=0,
        help="学习率余弦退火的总步数（0=沿用 epochs*iters）。"
        "只打算跑固定步数时必须显式给出，否则 lr 退火不完，loss 会在末尾降不下去",
    )

    # ========== 调试参数 ==========
    parser.add_argument(
        "--stop_step",
        default=0,
        type=int,
        help="跑到第几个 step 就停（0=不限制）；停止时会把训练状态落盘，便于续训",
    )

    # ========== MoE 专属参数 ==========
    # 【改动点】原来这里只有一行 `if args.use_moe: raise SystemExit(...)` 的拦截，
    # 拆掉拦截时把下面三个 argparse 参数一起删掉了，但 :286-288 仍在用它们，
    # 会导致 AttributeError。这里按 MiniMindConfig 的默认值补回。
    parser.add_argument(
        "--num_experts",
        default=4,
        type=int,
        help="MoE 的路由专家总数（只在 --use_moe 1 时生效）",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        default=1,
        type=int,
        help="每个 token 选几个专家（top-k 的 k）",
    )
    parser.add_argument(
        "--router_aux_loss_coef",
        default=5e-4,
        type=float,
        help="负载均衡损失系数；实测玩具模型上 5e-4 偏弱，真实 8 层模型够用",
    )

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # local_rank: 当前进程在本机上的GPU编号
    # 不同进程使用不同的种子，既保证随机性又保证可复现性
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        #MOE
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok, 
        router_aux_loss_coef=args.router_aux_loss_coef,
    )

    # 断点续训：加载之前的训练状态（模型、优化器、进度）
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR)
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    # bfloat16: 数值范围大，更稳定；float16: 更省内存但可能溢出
    # autocast 会自动选择精度，关键运算用 float32
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # CPU 不支持 autocast，用 nullcontext 作为空操作
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置 wandb 实验跟踪 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 未安装 swanlab 时降级为不记录，避免训练中断
        try:
            import swanlab as wandb  # pyright: ignore[reportMissingImports]
        except ImportError:
            wandb = None
            Logger("⚠️ 未安装swanlab，已跳过实验跟踪（可执行 pip install swanlab 启用）")

        if wandb is not None:
            # 有检查点时恢复同一个实验
            wandb_id = ckp_data.get("wandb_id") if ckp_data else None
            resume = "must" if wandb_id else None

            wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                id=wandb_id,
                resume=resume,
            )

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        tokenizer_path=DEFAULT_TOKENIZER_DIR,
        save_dir=args.save_dir,
        device=args.device,
    )

    # 模型 embedding 的行数（vocab_size）必须 >= 分词器词表大小，否则查表越界。
    # 反过来多几行是安全的（未使用的行不参与梯度），所以只在词表更大时才改模型。
    tok_vocab = len(tokenizer)
    if tok_vocab > lm_config.vocab_size:
        Logger(
            f"⚠️ 分词器词表({tok_vocab}) 大于模型 vocab_size({lm_config.vocab_size})，"
            f"已自动把模型词表调整到 {tok_vocab}"
        )
        resize_token_embeddings = getattr(model, "resize_token_embeddings", None)
        if callable(resize_token_embeddings):
            resize_token_embeddings(tok_vocab)
        else:
            # 自定义 MiniMind 模型中该属性可能是 embedding Tensor，不能直接调用。
            embedding = getattr(model, "tok_embeddings", None)
            embedding_attr = "tok_embeddings"
            if embedding is None:
                embedding = getattr(model, "embed_tokens", None)
                embedding_attr = "embed_tokens"
            if embedding is None:
                raise AttributeError("无法找到模型输入 embedding，无法扩展词表")

            old_embedding = embedding
            new_embedding = torch.nn.Embedding(
                tok_vocab,
                old_embedding.embedding_dim,
                device=old_embedding.weight.device,
                dtype=old_embedding.weight.dtype,
            )
            with torch.no_grad():
                new_embedding.weight[: old_embedding.num_embeddings].copy_(old_embedding.weight)
            setattr(model, embedding_attr, new_embedding)

            # 若输出层与输入 embedding 不共享权重，也同步扩展输出词表。
            lm_head = getattr(model, "lm_head", None)
            if isinstance(lm_head, torch.nn.Linear) and lm_head.out_features == old_embedding.num_embeddings:
                new_lm_head = torch.nn.Linear(
                    lm_head.in_features,
                    tok_vocab,
                    bias=lm_head.bias is not None,
                    device=lm_head.weight.device,
                    dtype=lm_head.weight.dtype,
                )
                with torch.no_grad():
                    new_lm_head.weight[: lm_head.out_features].copy_(lm_head.weight)
                    if lm_head.bias is not None:
                        new_lm_head.bias[: lm_head.out_features].copy_(lm_head.bias)
                model.lm_head = new_lm_head
        lm_config.vocab_size = tok_vocab
    elif tok_vocab < lm_config.vocab_size:
        Logger(
            f"ℹ️ 分词器词表({tok_vocab}) 小于模型 vocab_size({lm_config.vocab_size})，"
            f"embedding 多出的 {lm_config.vocab_size - tok_vocab} 行不会被用到（不影响训练）"
        )

    if args.packed:
        train_ds = PackedPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        Logger(
            f"📦 打包语料: {args.data_path} | {len(train_ds):,} 块 x {args.max_seq_len} | "
            f"token 利用率 {train_ds.meta.get('token_utilization_pct', '?')}% | "
            f"源: {train_ds.meta.get('src')} ({train_ds.meta.get('rows_read', 0):,} 行)"
        )
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])  # 恢复模型参数
        optimizer.load_state_dict(ckp_data["optimizer"])  # 恢复优化器状态（动量等）
        scaler.load_state_dict(ckp_data["scaler"])  # 恢复梯度缩放器状态
        start_epoch = ckp_data["epoch"]  # 恢复训练进度
        start_step = ckp_data.get("step", 0)
        Logger(
            f"⏩ 从 checkpoint 续训：epoch={start_epoch} step={start_step}"
            f"（权重与优化器状态均已恢复）"
        )

    # ---------- 续训护栏 ----------
    # 这里防的是 2026-09-21 实测到的那次「静默空转」：
    # checkpoint 的 step 恰好等于一整轮的步数（188/188），续训时 SkipBatchSampler
    # 跳过 188 个 batch 后什么都不剩，train_epoch 的 for 循环一次都不进，
    # 脚本却以退出码 0 正常结束 —— 看起来"训练完了"，实际 0 步。
    # 护栏：算清这一轮到底还有多少 batch 可跑，没得跑就明确报错，绝不静默成功。
    batches_per_epoch = (
        len(train_sampler) if train_sampler is not None else len(train_ds)
    ) // args.batch_size
    if ckp_data and start_step >= batches_per_epoch:
        has_next_epoch = start_epoch + 1 < args.epochs
        has_step_budget = args.total_steps > 0 and start_step < args.total_steps
        if not (has_next_epoch or has_step_budget):
            raise SystemExit(
                f"\n❌ 续训无事可做：checkpoint 记录 epoch={start_epoch} step={start_step}，"
                f"而本轮只有 {batches_per_epoch} 个 batch，且没有下一轮可跑。\n"
                f"   这通常意味着上一次训练已经跑完了最后一轮。\n"
                f"   要继续训练请二选一：\n"
                f"     · 加大 --epochs（例如 --epochs {start_epoch + 2}）\n"
                f"     · 用固定步数训练：--total_steps <大于 {start_step} 的数>\n"
                f"   不要指望 --from_resume 1 会自动多训一轮。\n"
            )
        Logger(
            f"⚠️ checkpoint 的 step({start_step}) 已达到本轮总 batch 数({batches_per_epoch})，"
            f"本轮数据已跑完，直接进入下一轮（配合 --total_steps 时按固定步数继续）"
        )
        start_step = 0
        start_epoch = start_epoch + 1  # 数据已跑完，推进到下一轮

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7. 开始训练 ==========
    interrupted = False
    for epoch in range(start_epoch, args.epochs):
        # 每个 epoch 设置不同的随机种子，确保数据顺序随机化
        if train_sampler:
            train_sampler.set_epoch(epoch)

        resume_this_epoch = epoch == start_epoch and start_step > 0

        if resume_this_epoch:  # 第一个epoch且存在检查点
            # 用跳批采样器跳过已训练的数据
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
                f"（本轮剩余 {len(loader)} 个 batch）"
            )
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}] 开始：{len(loader)} 个 batch，"
                f"batch_size={args.batch_size}，梯度累积={args.accumulation_steps}"
                f"（每 {args.accumulation_steps} 个 batch 更新一次参数）"
            )

        try:
            if resume_this_epoch:
                train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
            else:
                train_epoch(epoch, loader, len(loader), 0, wandb)
        except KeyboardInterrupt:
            # 手动打断（Ctrl+C）时把当前进度落盘，下次 --from_resume 1 能接着跑
            Logger("\n⚠️ 收到 Ctrl+C，正在保存训练状态以便续训……")
            save_state(epoch, _LAST_STEP[0], "KeyboardInterrupt")
            interrupted = True
            break

        # ---------- 一轮跑完：进度记为「下一个 epoch」----------
        # 这里就是静默空转的根因修复：原来存的是刚跑完的 epoch（0），
        # 续训时会被判成「这一轮还没跑完」，于是把整轮数据全部跳过。
        if is_main_process():
            save_state(epoch + 1, 0, f"epoch {epoch + 1} 完成")

    if interrupted:
        Logger("已中断退出；用同样的命令加 --from_resume 1 可继续训练。")


