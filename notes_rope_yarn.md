# freqs 与圈数 b 笔记

## 0 记号与数值（本项目参数）

| 符号 | 代码里 | 值 | 单位 |
|---|---|---|---|
| freqs[i] | `freqs` | — | rad/token |
| b | — | — | 无量纲（圈数） |
| B | `rope_base` | 1e6 | — |
| L | `original_max` | 2048 | token |
| d | `dim` | 64 | — |
| beta | `beta_fast` / `beta_slow` | 32 / 1 | 无量纲 |

---

## 1 一句话定义

freqs[i] 是"每 token 转多少弧度"；b 是"这一维在 L 内转了几圈"。
两者成正比，比例常数 2π/L —— 2π 来自"转一整圈 = 2π 弧度"，L 来自"在 L 这个窗口内计数"。

注意区分三个量（三者可互相换算，但单位和数值都不同）：

    freqs[i] = 角速度      单位 rad/token   第 13 维 ≈ 0.00365174
    lambda_i = 波长        单位 token       第 13 维 ≈ 1720.60   (转一圈要多少 token)
    b_i      = 圈数        单位 无          第 13 维 ≈ 1.190283   (在 L 内转了几圈)

关系：

    lambda_i = 2π / freqs[i] = L / b_i
    b_i      = L / lambda_i = (L / 2π) * freqs[i]

---

## 2 推导（得到 i 关于 beta 的公式）

前提：已知 freqs[i] = B^(-2i/d)，要求反函数 i = f(beta)。

第 1 步　位置与角度的关系

    theta = t * freqs[i]                     (rad)

第 2 步　在 L 个 token 内转了多少圈（定义式，无 log）

    b_i = theta_max / 2π = (L / 2π) * freqs[i]

第 3 步　把 freqs[i] 解出来（移项）

    freqs[i] = b_i / (L / 2π) = 2π * b_i / L

    建议写成 2π*b_i/L，不要写成 b_i/(L/2π)，双分式容易看错。

第 4 步　把 freqs[i] 展开，得到指数方程

    B^(-2i/d) = 2π * b_i / L

    量纲核对：左边无量纲；右边 b_i 无量纲、L 是 token，所以右边量纲 1/token，
    与 freqs 的量纲一致。

第 5 步　两边取自然对数（必须取对数，i 在指数上）

    (-2i/d) * lnB = ln(2π * b_i / L)

第 6 步　去负号、整理，解出 i

    (2i/d) * lnB = ln(L / (2π * b_i))

    i = (d / (2 * lnB)) * ln(L / (2π * beta))

其中 beta 是输入参数（圈数阈值），不是某个 b_i 的值。

---

## 3 数值验证（d=64, B=1e6, L=2048）

往返闭合，i -> b_i -> i：

| i | b_i = (L/2π)*freqs[i] | 代回 i(b_i) |
|---|---|---|
| 0 | 325.949323 | 0.0000 |
| 5 | 37.640041 | 5.0000 |
| 13 | 1.190283 | 13.0000 |
| 31 | 0.000502 | 31.0000 |

两个边界：

| beta | 来源配置项 | i(beta) | 含义 |
|---|---|---|---|
| 32 | beta_fast | 5.3760 | 高频边界：转过 32 圈以上，freqs 不缩放 |
| 1 | beta_slow | 13.4035 | 低频边界：转不满 1 圈，freqs 全量除以 factor |

---

## 4 落地要点与坑

### 4.1 方向：beta 越大，i 越小

    i(32) = 5.376  <  i(1) = 13.403

因为 beta = L/lambda，beta 大 = 波长短 = 高频 = 维度下标小。
所以拿到两个边界后必须交换，或直接用 min/max：

    low, high = min(low, high), max(low, high)

不交换的后果：gamma 的分母为负，clamp(0,1) 把所有维度压成 0 或 1，
整个 YaRN 静默失效，不报错。

### 4.2 必须用浮点边界，不要提前 int

    精确交点 i=13.4035 -> 该点圈数恰为 1.000000
    取整     i=13      -> 该维实际圈数 1.190

取整会把过渡带拉宽，gamma 每一维的权重都会偏。

### 4.3 必须 clamp(0, 1)

i 超出 [low, high] 时线性式会外推出负数或大于 1 的值，
gamma 是插值权重，物理上必须在 [0,1]。

### 4.4 括号（本式子最容易错的地方）

    dim / (2 * math.log(rope_base)) * math.log(original_max / (2 * math.pi * b))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    分母必须整体括起                     分母必须整体括起

    常见错误写法与后果：
      dim/2*log(B) * log(L/(2πb))   -> 除号只作用于 dim/2，结果偏大约 949 倍
      log(L / 2 * math.pi * b)      -> 按优先级是 (L/2)*π*b，beta=32 时偏 21 个维度

    更稳的写法（拆成多行，每行一次运算）：

    base_log  = math.log(rope_base)
    inv_scale = dim / (2 * base_log)                # d / (2 lnB)
    inv_dim   = lambda beta: inv_scale * math.log(original_max / (2 * math.pi * beta))

### 4.5 建议的断言

    i_fast, i_slow = inv_dim(beta_fast), inv_dim(beta_slow)   # 应为 5.376 / 13.403
    assert 5.0 < i_fast < 5.8,    f"beta_fast 边界错: {i_fast}"
    assert 13.1 < i_slow < 13.7,  f"beta_slow 边界错: {i_slow}"
    assert i_fast < i_slow,       "边界顺序反了"

断言要钉死具体数值区间，只写 assert i > 0 抓不住"静默算错"。

---

## 5 平文本版公式（贴不支持 LaTeX 的笔记用）

    （1）代码里已有的频率
        freqs[i] = B^(-2i/d)

    （2）圈数定义（正向，无 log）
        b_i = (L / 2π) * freqs[i]

    （3）移项，解出 freqs[i]
        freqs[i] = 2π * b_i / L

    （4）展开 freqs[i]，得到指数方程
        B^(-2i/d) = 2π * b_i / L

    （5）两边取自然对数
        (-2i/d) * lnB = ln(2π * b_i / L)

    （6）整理，解出 i（反向，有 log）
        i = (d / (2 * lnB)) * ln(L / (2π * beta))

    符号：d = dim, B = rope_base = 1e6, L = original_max = 2048,
          beta = beta_fast(32) 或 beta_slow(1)

    结果：i(32) = 5.376, i(1) = 13.403
