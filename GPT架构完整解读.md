# GPT 架构（Decoder-only Transformer）完整解读

> 基于代码 `GPT_transformer.py`，这是 **GPT 风格的纯解码器Transformer**

---

## 一句话概括

```
输入一段文字 -> 模型处理 -> 输出下一个最可能的字
```

---

## 完整架构图

```
输入: "今天天气"  (T=4)
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. 数据预处理 (第33-59行)                             │
│   红楼梦.txt -> 字符集 -> 字符↔整数映射 -> 训练/测试集                           │
│                                                                                 │
│   原始文本: "今天天气很好..."                                                    │
│       │                                                                        │
│       ▼                                                                        │
│   chars = ['，', '一', '不', '人', '今', '天', '好', ...]  (4244个不同字符)    │
│       │                                                                        │
│       ▼                                                                        │
│   stoi = {'，':0, '一':1, '不':2, ...}  (字符→整数)                             │
│   itos = {0:'，', 1:'一', 2:'不', ...}  (整数→字符)                             │
│       │                                                                        │
│       ▼                                                                        │
│   data = [0, 1, 2, 3, 4, ...]  (整数序列)                                      │
│       │                                                                        │
│       ▼                                                                        │
│   train_data(90%) + val_data(10%)                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           2. 数据批次获取 (第61-75行)                            │
│                                                                                 │
│   get_batch("train") -> 返回 xb, yb                                             │
│                                                                                 │
│   xb: 输入序列  (B=64, T=256)                                                   │
│       [[0,1,2,3,...,255], [5,8,12,...], ...]                                   │
│                                                                                 │
│   yb: 目标序列 (B=64, T=256)  <- 下一个字符                                    │
│       [[1,2,3,4,...,0], [8,12,16,...], ...]                                    │
│                                                                                 │
│   示例: xb[0] = "今天天气"                                                      │
│         yb[0] = "天天气?"  (预测下一个)                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           3. 嵌入层 (Embedding) 第175-202行                      │
│                                                                                 │
│  ┌───────────────────────┐        ┌───────────────────────┐                    │
│  │  Token Embedding      │        │  Position Embedding  │                    │
│  │  nn.Embedding        │        │  nn.Embedding       │                    │
│  │  (vocab_size, 384)   │        │  (block_size, 384) │                    │
│  │  (4244, 384)         │        │  (256, 384)        │                    │
│  └───────────┬───────────┘        └───────────┬───────────┘                    │
│              │                               │                                  │
│              ▼                               ▼                                  │
│  tok_emb: (B,T,384)              pos_emb: (T,384)                               │
│              │                               │                                  │
│              └───────────────┬───────────────┘                                  │
│                              ▼                                                  │
│                      x = tok_emb + pos_emb                                     │
│                      (B, T, 384)                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    4. 多级残差 Blocks (第180-183行) x n_layers=6              │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Block (代码152-168行) - Pre-Norm 结构                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                                                                 │   │   │
│  │  │  x ──► LayerNorm ──► MultiHeadAttention ──► +x (残差)         │   │   │
│  │  │       │                                                         │   │   │
│  │  │       │                                                         │   │   │
│  │  │       ▼                                                         │   │   │
│  │  │  LayerNorm ──► FeedForward ──► +x (残差)                       │   │   │
│  │  │                                                                 │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                              输入: (64, 256, 384)                         │   │
│  │                              输出: (64, 256, 384)                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Block 1-6 层层堆叠                                                      │   │
│  │  输入: (64, 256, 384) -> 输出: (64, 256, 384)                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  代码: self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layers)])│
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    5. 最终层归一化 + 输出层 (第205-209行)                       │
│                                                                                 │
│   x = self.ln_f(x)              # LayerNorm (最后归一化)                       │
│       │                                                                          │
│       ▼                                                                          │
│   logits = self.lm_head(x)      # Linear: 384 -> 4244                          │
│       │                                                                          │
│       ▼                                                                          │
│   输出: (B, T, vocab_size) = (64, 256, 4244)                                   │
│                                                                                 │
│   每个位置的每个字符都有一个"分数"                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           6. 损失计算 (第211-218行)                             │
│                                                                                 │
│   logits: (64*256, 4244)  <- 展平                                              │
│   targets: (64*256)                                                           │
│                                                                                 │
│   loss = cross_entropy(logits, targets)                                        │
│                                                                                 │
│   用途: 反向传播更新参数                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           7. 文本生成 (第222-247行)                             │
│                                                                                 │
│   generate(idx, max_new_tokens)                                                │
│                                                                                 │
│   for i in range(max_new_tokens):                                              │
│       1. 裁剪上下文 idx_cond = idx[:, -block_size:]                           │
│       2. forward得到logits                                                     │
│       3. 取最后一个位置的logits: logits[:, -1, :]                             │
│       4. softmax变成概率                                                        │
│       5. 采样下一个字符: torch.multinomial                                    │
│       6. 拼接到序列末尾                                                         │
│                                                                                 │
│   返回新生成的tokens                                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 各模块代码详解

### 模块1: Head (单头注意力) - 第81-121行

```python
class Head(nn.Module):
    """ 单头自注意力机制 (Self-Attention Head) """
    def __init__(self, head_size):
        super().__init__()
        # Key, Query, Value 线性变换
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # 注册下三角矩阵用于因果掩码
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 计算注意力分数
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        # 因果掩码
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        # Softmax + Dropout
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 加权求和
        v = self.value(x)
        out = wei @ v
        return out
```

**作用**: 让每个字"看到"前面的字

---

### 模块2: MultiHeadAttention (多头注意力) - 第123-136行

```python
class MultiHeadAttention(nn.Module):
    """ 多头注意力机制：多个 Head 并行计算并拼接 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

**作用**: 8个头关注不同关系（语法、语义、位置等）

---

### 模块3: FeedForward (前馈网络) - 第138-150行

```python
class FeedForward(nn.Module):
    """ 前馈神经网络 (MLP)：逐位置进行非线性变换 """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 升维
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # 降维
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
```

**作用**: 每个字单独"消化"信息，增强表达能力

---

### 模块4: Block (残差块) - 第152-168行

```python
class Block(nn.Module):
    """ Transformer Block: 通讯 (Attention) + 计算 (FeedForward) """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-Norm 结构
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

**作用**: 6层堆叠，层层提取语义

---

### 模块5: LanguageModel (主模型) - 第170-220行

```python
class LanguageModel(nn.Module):
    """ 完整的字符级 GPT 模型 """
    def __init__(self):
        super().__init__()
        
        # 1. 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 2. 位置嵌入
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        
        # 3. Transformer Blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads) for _ in range(n_layers)]
        )
        
        # 4. 最终的 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 5. 输出头
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.to(device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

---

## 维度变化总览

```
阶段                    形状                   说明
──────────────────────────────────────────────────────────────────
输入 idx              (64, 256)              字符索引
Token Embedding      (64, 256, 384)          词嵌入
+ Position Embedding (64, 256, 384)          加上位置
                          │
    ┌─────────────────────┴─────────────────────┐
    │           n_layers=6 Block                │
    │   Pre-Norm:                               │
    │   x + sa(ln1(x))                          │
    │   x + ffwd(ln2(x))                        │
    └───────────────────────────────────────────┘
                          │
最终 LayerNorm         (64, 256, 384)
lm_head (Linear)      (64, 256, 4244)         预测分数
```

---

## 训练流程 (第252-304行)

```
main():
    1. 创建模型
    2. 优化器 AdamW
    3. 循环 1000 次:
        - get_batch("train")
        - forward 计算 loss
        - zero_grad + backward + step
        - 每50轮打印损失
    4. 生成文本:
        - 随机选一段上文
        - 获取原文真实续写(Ground Truth)
        - 用模型续写500字
        - 对比输出: 原文 vs 模型生成
```

---

## 超参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 64 | 并行处理64条序列 |
| block_size | 256 | 每条序列256个字 |
| n_embd | 384 | 嵌入维度 |
| num_heads | 8 | 8个注意力头 |
| head_size | 48 | 每头48维 (384÷8) |
| n_layers | 6 | 6层Block |
| vocab_size | 4244 | 字符种类数 |
| dropout | 0.2 | 丢弃率 |
| learning_rate | 3e-4 | 学习率 |
| max_iters | 1000 | 训练轮数 |

---

## Pre-Norm 结构详解

```
Pre-Norm (你的代码):
    x = x + Sublayer(LayerNorm(x))
    优点: 更稳定，训练更容易
```

---

## 总结

```
你的代码 = GPT-2 风格的 纯解码器Transformer

输入: "今天天气"
   │
   ▼
[词嵌入 + 位置嵌入] -> [Block x 6] -> [输出]
   │                      │
   │                      ├─ Pre-Norm 结构
   │                      ├─ 多头注意力(8头)
   │                      ├─ 残差连接(2个)
   │                      ├─ 前馈网络(4倍维度)
   │                      └─ 层层堆叠提取语义
   │
   ▼
输出: "很好" (预测下一个字)
```

**一句话**: 这是一个基于Transformer解码器的语言模型，通过多头注意力机制让每个字符看到序列中前面的所有字符，从而预测下一个最可能的字符。

---

## 代码特点

1. **清晰的分章节注释** - `# ====` 分隔各部分
2. **完整的 docstring** - 每个类都有说明
3. **相对路径处理** - 支持不同机器运行
4. **Pre-Norm 结构** - 更稳定的训练
5. **高效的梯度清零** - `set_to_none=True`
6. **对比输出** - 显示原文vs模型生成，方便评估
