import os, time, argparse, warnings
import torch
from transformers import AutoTokenizer, TextStreamer
from model.model import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

# 预训练权重只会"续写"，所以给的是开头片段，不是"问题"
PRESETS = {
    'news':      ['人工智能是', '根据最新研究报告显示', '记者昨日从有关部门获悉', '今天天气'],
    'knowledge': ['中国的首都是', '水在标准大气压下的沸点是', '光合作用的本质是', '牛顿第一定律指出'],
    'poem':      ['床前明月光', '春风又绿江南岸', '大江东去浪淘尽'],
}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='MiniMind 预训练权重续写')
    parser.add_argument('--load_from', default='model', type=str, help='分词器目录')
    parser.add_argument('--save_dir', default='out', type=str, help='权重目录')
    parser.add_argument('--weight', default='pretrain', type=str, help='权重前缀（不含 _512 后缀）')
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    # 【改动点】加 --use_moe：MoE 存档名带 _moe 后缀（pretrain_moe_512_moe.pth），
    # 且模型必须以 use_moe=True 构造，否则 strict=True 加载会 size mismatch。
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help='1=加载 MoE 权重（存档名带 _moe 后缀）')
    parser.add_argument('--max_new_tokens', default=80, type=int)
    parser.add_argument('--temperature', default=0.9, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--preset', default='news', choices=list(PRESETS))
    parser.add_argument('--prompt', default='', type=str, help='自定义提示词，非空则覆盖 preset')
    parser.add_argument('--show_speed', default=1, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.load_from, local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)))          # 【改动点】跟随 --use_moe 构造
    # 【改动点】与 train_pretrain.py:159-160 用同一套后缀规则，否则拼不出 MoE 存档名
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')
    model.load_state_dict(torch.load(ckp, map_location='cpu', weights_only=True), strict=True)
    model = model.half().eval().to(args.device)      # 官方这行是对的，保留
    print(f'已加载 {ckp} | {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M\n')

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    for prompt in ([args.prompt] if args.prompt else PRESETS[args.preset]):
        print(f'\n片段: {prompt}')
        setup_seed(2026)
        # 预训练模型只认"纯文本前缀"，不走 chat template
        print('续写: ', end='')
        st = time.time()
        inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors='pt').to(args.device)
        generated_ids = model.generate(
            inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1,
        )
        n = len(generated_ids[0]) - len(inputs['input_ids'][0])
        if args.show_speed:
            print(f'\n[Speed]: {n / (time.time() - st):.2f} tokens/s')


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()