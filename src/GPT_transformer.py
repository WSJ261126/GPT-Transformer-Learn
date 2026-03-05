# 手搓Transformer
import os
import random
import textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. 超参数设置
# ==============================================================================
batch_size = 64        # 批次大小：同时并行处理多少条数据
block_size = 256       # 上下文长度：训练/验证时的序列最大长度
n_embd = 384           # 嵌入维度：每个token被表示为多少维的向量
num_heads = 8          # 注意力头数
head_size = n_embd // num_heads  # 每个头的维度 (384 // 8 = 48)
n_layers = 6           # Transformer Block 层数
dropout = 0.2          # Dropout 概率
learning_rate = 3e-4   # 学习率
max_iters = 5000       # 最大训练迭代次数
eval_interval = 200    # 每多少轮评估一次损失
eval_iters = 200       # 评估时采样的 batch 数量
width = 60             # 文本生成时的换行宽度

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# 数据文件路径 
file_name = os.path.join(os.path.dirname(__file__), '..', 'data', '红楼梦.txt')
if not os.path.exists(file_name):
    # 尝试在当前脚本所在目录查找
    file_name = os.path.join(os.path.dirname(__file__), "红楼梦.txt")

# ==============================================================================
# 2. 数据预处理
# ==============================================================================
print(f"正在读取文件: {file_name} ...")
with open(file_name, 'r', encoding="utf-8") as f:
    text = f.read()

# 构建词表：所有出现的字符去重并排序
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符 <-> 整数 映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]           # 编码: str -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])  # 解码: list[int] -> str

# 划分数据集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

print(f"词汇表大小: {vocab_size}")
print(f"数据总长度: {len(data)}")
print(f"训练集/验证集长度: {len(train_data)} / {len(val_data)}")
print(f"运行设备: {device}")

def get_batch(split):
    """
    获取一个批次的训练/验证数据
    x: 输入序列 (batch_size, block_size)
    y: 目标序列 (batch_size, block_size)，即 x 向后移一位
    """
    data = train_data if split == "train" else val_data
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """
    在训练集和验证集上各采样 eval_iters 个 batch，返回平均 loss。
    使用 no_grad + eval 模式：结果更稳定，不消耗梯度显存。
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==============================================================================
# 3. 模型定义
# ==============================================================================

class Head(nn.Module):
    """ 单头自注意力机制 (Self-Attention Head) """
    def __init__(self, head_size):
        super().__init__()
        # Key, Query, Value 线性变换
        # 这里的线性层没有 bias，纯粹是线性映射
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 注册一个下三角矩阵作为 buffer，不作为模型参数更新
        # 用于 masked attention，防止看到未来的信息
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入维度: (Batch, Time, Channel/Embedding)
        B, T, C = x.shape

        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # 计算注意力分数 (Attention Scores)
        # wei = q @ k.T / sqrt(head_size)
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Masking: 将上三角部分（未来信息）设为 -inf，softmax 后变为 0
        # 使用切片 [:T, :T] 以适配生成过程中序列长度小于 block_size 的情况
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax 归一化
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 计算加权后的值
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    """ 多头注意力机制：多个 Head 并行计算并拼接 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 拼接所有头的输出: (B, T, head_size) * num_heads -> (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 最后的线性投影和 dropout
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ 前馈神经网络 (MLP)：逐位置进行非线性变换 """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 升维 (通常是 4 倍)
            nn.GELU(),    # GPT-2 使用 GELU，中文任务上效果优于 ReLU
            nn.Linear(4 * n_embd, n_embd), # 降维回 n_embd
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: 通讯 (Attention) + 计算 (FeedForward) """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        # LayerNorm 层
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-Norm 结构：先 Norm 再 Attention/FFWD
        # 残差连接 (Residual Connection): x + ...
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    """ 完整的字符级 GPT 模型 """
    def __init__(self):
        super().__init__()

        # 1. 词嵌入 (Token Embedding)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 2. 位置嵌入 (Position Embedding)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        # 3. Transformer Blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads) for _ in range(n_layers)]
        )

        # 4. 最终的 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # 5. 输出头 (Language Model Head)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 权重初始化（GPT-2 风格：均值0，标准差0.02 的正态分布）
        self.apply(self._init_weights)

        # 移动到设备
        self.to(device)

    def _init_weights(self, module):
        """ GPT-2 风格权重初始化，比 PyTorch 默认初始化收敛更快 """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 获取 Token Embedding 和 Position Embedding
        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        pos_emb = self.pos_embedding(torch.arange(T, device=device)) # (T, n_embd)

        # 融合信息
        x = tok_emb + pos_emb # (B, T, n_embd)

        # 通过 Transformer Blocks
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x)   # (B, T, n_embd)

        # 计算 Logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape 为 (B*T, vocab_size) 以适配 cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        自回归生成文本
        idx: 当前上下文 (B, T)
        max_new_tokens: 生成多少个新字符
        使用 @torch.no_grad() 禁用梯度计算，节省显存
        """
        for _ in range(max_new_tokens):
            # 裁剪上下文，确保不超过 block_size
            idx_cond = idx[:, -block_size:]

            # 前向传播，获取预测
            logits, _ = self.forward(idx_cond)

            # 关注最后一个时间步
            logits = logits[:, -1, :] # (B, C)

            # 计算概率
            probs = F.softmax(logits, dim=-1) # (B, C)

            # 采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, -max_new_tokens:] # 只返回新生成的 tokens

# ==============================================================================
# 4. 主函数
# ==============================================================================
def main():
    print("-" * 50)
    print("开始训练...")

    model = LanguageModel()

    # 打印参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(max_iters):
        # 获取一个 batch
        xb, yb = get_batch("train")

        # 前向传播
        logits, loss = model(xb, yb)

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：防止深层模型训练时梯度爆炸导致 loss 变 nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 定期评估训练集和验证集的平均损失
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {i:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # 保存模型
    save_path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存至: {save_path}")

    print("\n训练结束，开始生成文本...")
    print("-" * 50)

    # 随机选择一段验证集数据作为开头
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens - 1)
    context = val_data[start_idx : start_idx + block_size].unsqueeze(0).to(device) # (1, T)

    # 获取真实的后续文本 (Ground Truth)
    # GPT 是生成式模型，目标是"模仿风格"而非"背诵原文"，此处仅作对比参考
    real_next_tokens = val_data[start_idx + block_size : start_idx + block_size + max_new_tokens]
    real_next_str = decode(real_next_tokens.tolist())

    # 切换到评估模式再生成（关闭 dropout，结果更稳定）
    model.eval()
    generated_tokens = model.generate(context, max_new_tokens)
    model.train()

    # 解码并打印
    input_str = decode(context[0].tolist())
    generated_str = decode(generated_tokens[0].tolist())

    print("【输入上下文】:")
    print(textwrap.fill(input_str, width=width))
    print("\n【原文真实续写】(标准答案):")
    print(textwrap.fill(real_next_str, width=width))
    print("\n【模型生成续写】(模型创作):")
    print(textwrap.fill(generated_str, width=width))
    print("-" * 50)

if __name__ == "__main__":
    main()
