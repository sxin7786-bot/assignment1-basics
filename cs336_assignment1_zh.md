# CS336 作业1（基础篇）：从零构建 Transformer 语言模型

> **版本**：26.0.3  
> **作者**：CS336 课程组  
> **学期**：2026年春季

---

## 1. 作业概览

本作业要求你从零开始构建训练标准 Transformer 语言模型（LM）所需的全部组件，并完成模型训练。

### 需要实现的内容

1. 字节对编码（BPE）分词器（第2节）
2. Transformer 语言模型（第3节）
3. 交叉熵损失函数与 AdamW 优化器（第4节）
4. 训练循环，支持模型和优化器状态的序列化与加载（第5节）

### 需要运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器
2. 使用训练好的分词器对数据集进行编码，转换为整数 ID 序列
3. 在 TinyStories 数据集上训练 Transformer 语言模型
4. 使用训练好的模型生成样本并评估困惑度（Perplexity）
5. 在 OpenWebText 上训练模型，并将所得困惑度提交至排行榜

### 可以使用的内容

每个组件需从零实现。特别地，**不得**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的定义，以下情况除外：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（如 `Module`、`ModuleList`、`Sequential` 等）
- `torch.optim.Optimizer` 基类

其他 PyTorch 定义均可使用。如不确定某函数是否被允许，请在 Slack 上提问。

### 关于 AI 工具的声明

AI 工具可用于回答高层次概念性问题，或提供底层编程文档（如函数签名和库 API）。**但不允许使用 AI 工具实现作业的任何部分**，包括编程智能体（如 Cursor Agents、Codex、Claude Code）和 AI 自动补全（如 Cursor Tab、GitHub Copilot）。

强烈建议在完成作业时禁用 IDE 中的 AI 自动补全。

### 代码结构

代码仓库地址：[github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

- `cs336_basics/*`：你编写代码的目录（从零开始）
- `adapters.py`：适配器文件，将你的代码接口暴露给测试（仅填写调用逻辑，不含实质逻辑）
- `test_*.py`：所有测试文件（不得修改）

### 提交方式

运行 `make_submission.sh` 生成提交 zip 文件，提交以下内容至 Gradescope：

- `writeup.pdf`：所有书面问题的回答（请使用排版工具）
- `code.zip`：所有编写的代码

排行榜提交请向以下仓库提交 PR：[github.com/stanford-cs336/assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)

### 数据集获取

本作业使用两个预处理数据集：TinyStories 和 OpenWebText，均为单个大型纯文本文件。具体下载方式详见仓库 README.md。

> **低资源提示**：在课程讲义中，我们会给出在资源有限（少量或没有 GPU）的情况下完成作业的建议，例如缩小数据集或模型规模，或在 Mac 集成显卡或 CPU 上运行训练代码。即使你是有权使用课程机器的斯坦福在读生，这些提示也可能帮助你更快迭代。

> **Apple Silicon / CPU 提示**：使用参考代码，可在 Apple M4 Max 芯片（36GB RAM）上，于 Metal GPU（MPS）上不到5分钟、CPU 上约30分钟内训练出能生成较流畅文本的语言模型。

---

## 2. 字节对编码（BPE）分词器

本部分将训练并实现一个字节级的字节对编码（BPE）分词器。我们将把任意 Unicode 字符串表示为字节序列，并在此字节序列上训练 BPE 分词器，之后用它将文本（字符串）编码为用于语言建模的词元（整数序列）。

### 2.1 Unicode 标准

Unicode 是将字符映射到整数码点的文本编码标准。截至 Unicode 17.0（2025年9月发布），标准共定义了覆盖172种文字的159,801个字符。例如，字符 `s` 的码点为 115（U+0073），字符 `牛` 的码点为 29275。

在 Python 中，可使用 `ord()` 将单个 Unicode 字符转为整数，`chr()` 将整数码点转为字符。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

#### 问题（unicode1）：理解 Unicode（1分）

**(a)** `chr(0)` 返回什么 Unicode 字符？  
**交付物**：一句话回答。

**(b)** 该字符的字符串表示（`__repr__()`）与打印表示有何不同？  
**交付物**：一句话回答。

**(c)** 当该字符出现在文本中时会发生什么？可尝试在 Python 解释器中运行以下代码：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**交付物**：一句话回答。

---

### 2.2 Unicode 编码

Unicode 标准定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上最主流的编码（超过98%的网页使用）。

在 Python 中，使用 `encode()` 将 Unicode 字符串编码为 UTF-8，使用 `list()` 获取底层字节值，使用 `decode()` 将 UTF-8 字节字符串解码回 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, ...]
```

#### 问题（unicode2）：Unicode 编码（3分）

**(a)** 相比 UTF-16 或 UTF-32，为什么更倾向于在 UTF-8 编码的字节上训练分词器？  
**交付物**：一到两句话回答。

**(b)** 下面这个函数意图将 UTF-8 字节字符串解码为 Unicode 字符串，但实现有误。为什么？请给出一个产生错误结果的输入示例：

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

**交付物**：一个示例输入字节字符串，及一句话解释原因。

**(c)** 给出一个无法解码为任何 Unicode 字符的两字节序列。  
**交付物**：示例及一句话解释。

---

### 2.3 子词分词

子词分词是词级分词器和字节级分词器之间的折中方案。字节级分词器词表有 256 个条目，而子词分词器以更大的词表换取更好的输入字节序列压缩率。

字节对编码（BPE）是一种压缩算法，通过迭代地将最高频的字节对替换（"合并"）为一个新的未使用索引来构建词表。使用 BPE 构建词表的子词分词器通常称为 BPE 分词器。

---

### 2.4 BPE 分词器训练

BPE 分词器训练包含以下三个主要步骤：

#### 词表初始化

由于我们训练的是字节级 BPE 分词器，初始词表即所有字节的集合，共 256 个。

#### 预分词（Pre-tokenization）

预分词是对语料库的粗粒度分词，帮助统计字符对的出现频率。我们使用类似 GPT-2 的基于正则表达式的预分词器：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

示例：

```python
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

实际代码中应使用 `re.finditer` 而非 `re.findall`，以避免在构建频率映射时存储所有预词元。

#### 计算 BPE 合并

1. 统计所有字节对的频率，找出频率最高的字节对 `("A", "B")`
2. 将所有 `("A", "B")` 合并为新词元 `"AB"`，加入词表
3. 最终词表大小 = 初始词表大小（256）+ 合并操作次数

**平局处理**：若多对字节具有相同最高频率，优先选择字典序更大的那对。

#### 特殊词元（Special Tokens）

某些字符串（如 `<|endoftext|>`）用于编码元数据（如文档边界）。这类字符串应作为"特殊词元"，永远不会被拆分为多个词元。

#### 示例（bpe_example）：BPE 训练示例

语料库：

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

- 初始词表包含特殊词元 `<|endoftext|>` 和256个字节值
- 预分词（按空格分割）后得到频率表：`{low:5, lower:2, widest:3, newest:6}`
- 第一轮：统计字节对频率，`(e,s)` 和 `(s,t)` 并列最高，取字典序更大的 `(s,t)` 进行合并
- 经过6次合并后，词表新增：`st, est, ow, low, west, ne`
- 用该词表，`newest` 被分词为 `[ne, west]`

---

### 2.5 实验：在 TinyStories 上训练 BPE 分词器

#### 并行化预分词

预分词是主要瓶颈，可使用内置 `multiprocessing` 库并行化。建议在特殊词元起始处对语料库进行分块，可直接使用以下参考代码：

```
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
```

#### 预分词前移除特殊词元

在运行正则预分词前，应从语料库中剔除所有特殊词元，并以特殊词元为分割点，确保合并不会跨越文档边界。

#### 优化合并步骤

可通过索引所有字节对的计数并在每次合并后增量更新，而非每次都遍历所有字节对，显著加速 BPE 训练。

> **低资源提示**：建议先在 TinyStories 验证集（22K 文档而非 220万文档）上训练，以加快开发迭代速度。

#### 问题（train_bpe）：BPE 分词器训练（15分）

**交付物**：实现一个函数，给定输入文本文件路径，训练一个（字节级）BPE 分词器。

**输入参数**：
- `input_path: str` — 训练数据文本文件路径
- `vocab_size: int` — 最大词表大小（含初始字节词表、合并产生的词元和特殊词元）
- `special_tokens: list[str]` — 需加入词表的特殊字符串列表

**输出**：
- `vocab: dict[int, bytes]` — 词表，映射词元 ID 到字节
- `merges: list[tuple[bytes, bytes]]` — 按创建顺序排列的 BPE 合并列表

实现测试适配器 `[adapters.run_train_bpe]`，运行 `uv run pytest tests/test_train_bpe.py`。

#### 问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2分）

**(a)** 在 TinyStories 数据集上训练最大词表大小为 10,000 的字节级 BPE 分词器，添加 `<|endoftext|>` 特殊词元。训练耗时和内存是多少？词表中最长的词元是什么？是否合理？  
**资源要求**：≤30分钟（无 GPU），≤30GB RAM  
**交付物**：一到两句话回答。

**(b)** 对代码进行性能分析。分词器训练过程中哪个部分耗时最多？  
**交付物**：一到两句话回答。

#### 问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2分）

**(a)** 在 OpenWebText 数据集上训练最大词表大小为 32,000 的字节级 BPE 分词器。词表中最长的词元是什么？是否合理？  
**资源要求**：≤12小时（无 GPU），≤100GB RAM  
**交付物**：一到两句话回答。

**(b)** 比较 TinyStories 和 OpenWebText 训练所得分词器的异同。  
**交付物**：一到两句话回答。

---

### 2.6 BPE 分词器：编码与解码

#### 2.6.1 编码文本

**步骤1：预分词**  
对文本进行预分词，将每个预词元表示为 UTF-8 字节序列。

**步骤2：应用合并**  
按训练时的合并创建顺序，将合并规则依次应用到每个预词元。

##### 示例（bpe_encoding）：BPE 编码示例

- 输入字符串：`'the cat ate'`
- 预分词结果：`['the', ' cat', ' ate']`
- 逐步应用合并规则，最终编码结果：`[9, 7, 1, 5, 10, 3]`

#### 2.6.2 解码文本

将整数词元 ID 序列解码回原始文本：查找每个 ID 对应的字节序列，拼接后用 `bytes.decode(errors='replace')` 解码为 Unicode 字符串（无效字节用 U+FFFD 替换）。

#### 问题（tokenizer）：实现分词器（15分）

**交付物**：实现一个 `Tokenizer` 类，支持：

```python
def __init__(self, vocab, merges, special_tokens=None)
    # vocab: dict[int, bytes]
    # merges: list[tuple[bytes, bytes]]
    # special_tokens: list[str] | None = None

@classmethod
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
    # 从文件加载词表和合并规则

def encode(self, text: str) -> list[int]
    # 将文本编码为词元 ID 序列

def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
    # 懒式编码，支持大文件流式处理

def decode(self, ids: list[int]) -> str
    # 将词元 ID 序列解码为文本
```

实现测试适配器 `[adapters.get_tokenizer]`，运行 `uv run pytest tests/test_tokenizer.py`。

---

### 2.7 实验

#### 问题（tokenizer_experiments）：分词器实验（4分）

**(a)** 从 TinyStories 和 OpenWebText 各抽取10个文档，分别用对应分词器编码，计算压缩率（字节/词元）。  
**交付物**：一到两句话回答。

**(b)** 用 TinyStories 分词器对 OpenWebText 样本进行分词会发生什么？对比压缩率并定性描述结果。  
**交付物**：一到两句话回答。

**(c)** 估算分词器的吞吐量（字节/秒）。对 Pile 数据集（825GB 文本）进行分词需要多久？  
**交付物**：一到两句话回答。

**(d)** 将训练集和验证集编码为整数词元 ID 序列。建议使用 `uint16` 数据类型的 NumPy 数组序列化，为什么 `uint16` 是合适的选择？  
**交付物**：一到两句话回答。

---

## 3. Transformer 语言模型架构

语言模型以批次化的整数词元 ID 序列（形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`）为输入，返回对词表的归一化概率分布（形状为 `(batch_size, sequence_length, vocab_size)` 的张量）。

### 3.1 Transformer LM

给定词元 ID 序列，Transformer 语言模型使用：
1. **词嵌入层**：将词元 ID 转换为稠密向量
2. **若干个 Transformer 块**：聚合序列信息并非线性变换
3. **线性投影（LM Head）**：产生预测的下一词元 logits

（参见图1：Transformer 语言模型总览；图2：Pre-norm Transformer 块）

### 3.2 关于批处理、Einsum 和高效计算

Transformer 中很多操作需要批量处理多个输入。推荐学习并使用 `einsum` 符号（通过 `torch.einsum` 或框架无关的 `einops`/`einx` 库），以获得更好的可读性和灵活性。

**关键操作**：
- `einsum`：可对任意维度输入张量做张量缩并
- `rearrange`：可对任意维度进行重排、拼接和拆分

#### 示例（einstein_example1）：用 einops.einsum 做批矩阵乘法

```python
from einops import rearrange, einsum

# 自文档化且鲁棒
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

# 支持任意前导批维度
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

#### 3.2.1 数学符号与内存排列

本作业数学部分主要使用列向量，线性变换为 $y = Wx$。注意 PyTorch 使用行优先（row-major）内存排列，在用普通矩阵乘法时需对矩阵转置。使用 `einsum` 符号可避免此问题。

---

### 3.3 基础模块：Linear 与 Embedding

#### 3.3.1 参数初始化

- **线性层权重**：$\mathcal{N}(\mu=0, \sigma^2=\frac{2}{d_{in}+d_{out}})$，截断于 $[-3\sigma, 3\sigma]$
- **嵌入层**：$\mathcal{N}(\mu=0, \sigma^2=1)$，截断于 $[-3, 3]$
- **RMSNorm**：初始化为全1

使用 `torch.nn.init.trunc_normal_` 初始化截断正态权重。

#### 3.3.2 Linear 模块

线性变换：$y = Wx$（无偏置项，遵循大多数现代 LLM 的做法）

#### 问题（linear）：实现 Linear 模块（1分）

**交付物**：实现继承自 `torch.nn.Module` 的 `Linear` 类：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor
```

注意：
- 子类化 `nn.Module`
- 调用父类构造函数
- 将参数存储为 $W$（而非 $W^\top$），放入 `nn.Parameter`
- 不得使用 `nn.Linear` 或 `nn.functional.linear`

实现测试适配器 `[adapters.run_linear]`，运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

嵌入层将整数词元 ID 映射到维度为 `d_model` 的向量空间。

#### 问题（embedding）：实现 Embedding 模块（1分）

**交付物**：实现继承自 `torch.nn.Module` 的 `Embedding` 类：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

注意：
- 不得使用 `nn.Embedding` 或 `nn.functional.embedding`
- 嵌入矩阵的最后一维为 `d_model`

实现测试适配器 `[adapters.run_embedding]`，运行 `uv run pytest -k test_embedding`。

---

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块包含两个子层：**多头自注意力机制**和**位置前馈网络**。

与原始 Transformer 的 post-norm 不同，pre-norm 将层归一化移至每个子层的输入端（另在最后一个 Transformer 块后额外加一层归一化），以提升训练稳定性。这种 pre-norm Transformer 是当今语言模型的标准（如 GPT-3、LLaMA、PaLM 等）。

#### 3.4.1 均方根层归一化（RMSNorm）

给定激活向量 $a \in \mathbb{R}^{d_\text{model}}$，RMSNorm 将每个激活 $a_i$ 重缩放为：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_\text{model}} \sum_{i=1}^{d_\text{model}} a_i^2 + \varepsilon}$，$g_i$ 是可学习的增益参数，$\varepsilon$ 通常固定为 `1e-5`。

实现时应将输入上转型为 `torch.float32` 以防止溢出：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# ... RMSNorm 计算 ...
return result.to(in_dtype)
```

#### 问题（rmsnorm）：实现 RMSNorm（1分）

**交付物**：实现 `RMSNorm` 为 `torch.nn.Module`：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
def forward(self, x: torch.Tensor) -> torch.Tensor  # 形状: (batch_size, seq_len, d_model)
```

实现测试适配器 `[adapters.run_rmsnorm]`，运行 `uv run pytest -k test_rmsnorm`。

---

#### 3.4.2 位置前馈网络（SwiGLU）

现代语言模型对原始 Transformer 的前馈网络进行了两项主要改进：使用不同的激活函数，以及引入门控机制。

**SiLU（Swish）激活函数**：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**门控线性单元（GLU）**：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x$$

**SwiGLU**（将 SiLU 和 GLU 结合）：

$$\text{FFN}(x) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)$$

其中 $x \in \mathbb{R}^{d_\text{model}}$，$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$，$W_2 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$，$d_{ff} = \frac{8}{3} d_\text{model}$（实现时取最近的64的倍数）。

#### 问题（positionwise_feedforward）：实现位置前馈网络（2分）

**交付物**：实现 SwiGLU 前馈网络。

实现测试适配器 `[adapters.run_swiglu]`，运行 `uv run pytest -k test_swiglu`。

---

#### 3.4.3 旋转位置编码（RoPE）

为注入位置信息，我们实现旋转位置编码（RoPE）。对于位置 $i$ 处的查询词元 $q^{(i)} \in \mathbb{R}^d$，应用旋转矩阵 $R_i$，其中 $R_i$ 为分块对角矩阵，每个 $2\times 2$ 的旋转块 $R_i^k$ 为：

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}$$

旋转角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$，$\Theta$ 为常数（如10000）。

该层没有可学习参数；$\cos$ 和 $\sin$ 值可预计算并通过 `self.register_buffer(persistent=False)` 存储。

#### 问题（rope）：实现 RoPE（2分）

**交付物**：实现 `RotaryPositionalEmbedding` 类：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
    # x 形状: (..., seq_len, d_k)
    # token_positions 形状: (..., seq_len)
```

实现测试适配器 `[adapters.run_rope]`，运行 `uv run pytest -k test_rope`。

---

#### 3.4.4 缩放点积注意力

**Softmax**：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^n \exp(v_j)}$$

为数值稳定性，使用减最大值技巧（不改变结果）。

#### 问题（softmax）：实现 Softmax（1分）

**交付物**：实现 softmax 函数，支持指定维度，使用减最大值技巧。

实现适配器 `[adapters.run_softmax]`，运行 `uv run pytest -k test_softmax_matches_pytorch`。

**注意力操作**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

支持可选的布尔掩码 $M$（True 表示可访问，False 表示不可访问，通过在 softmax 前将 False 位置设为 $-\infty$ 实现）。

#### 问题（scaled_dot_product_attention）：实现缩放点积注意力（5分）

**交付物**：实现缩放点积注意力函数，支持形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，及可选布尔掩码。

实现测试适配器 `[adapters.run_scaled_dot_product_attention]`，运行 `uv run pytest -k test_scaled_dot_product_attention`。

---

#### 3.4.5 因果多头自注意力

**多头注意力**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

**多头自注意力**：

$$\text{MultiHeadSelfAttention}(x) = W^O \text{MultiHead}(W^Q x, W^K x, W^V x)$$

其中 $d_k = d_v = \frac{d_\text{model}}{h}$。

**因果掩码**：使用下三角掩码，允许词元 $i$ 仅关注位置 $j \leq i$ 的词元，防止模型访问未来词元（可使用 `torch.triu` 或广播索引比较构造）。

**应用 RoPE**：对查询和键向量（而非值向量）应用 RoPE，多头维度作为批维度处理。

#### 问题（multihead_self_attention）：实现因果多头自注意力（5分）

**交付物**：实现因果多头自注意力为 `torch.nn.Module`，至少接受以下参数：
- `d_model: int`
- `num_heads: int`

实现测试适配器 `[adapters.run_multihead_self_attention]`，运行 `uv run pytest -k test_multihead_self_attention`。

---

### 3.5 完整 Transformer LM

#### 问题（transformer_block）：实现 Transformer 块（3分）

**交付物**：实现 pre-norm Transformer 块：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x))$$

接受参数：`d_model`、`num_heads`、`d_ff`。

实现适配器 `[adapters.run_transformer_block]`，运行 `uv run pytest -k test_transformer_block`。

#### 问题（transformer_lm）：实现 Transformer LM（3分）

**交付物**：将所有组件组合成完整的 Transformer 语言模型（词嵌入 → N个 Transformer 块 → 最终 RMSNorm → LM Head）。额外参数：
- `vocab_size: int`
- `context_length: int`
- `num_layers: int`

实现测试适配器 `[adapters.run_transformer_lm]`，运行 `uv run pytest -k test_transformer_lm`。

---

### 资源核算

矩阵乘法规则：给定 $A \in \mathbb{R}^{m\times n}$，$B \in \mathbb{R}^{n\times p}$，乘积 $AB$ 需要 $2mnp$ FLOPs。

#### 问题（transformer_accounting）：Transformer LM 资源核算（5分）

考虑 GPT-2 XL 规模的模型：

| 参数 | 值 |
|------|-----|
| vocab_size | 50,257 |
| context_length | 1,024 |
| num_layers | 48 |
| d_model | 1,600 |
| num_heads | 25 |
| d_ff | 4,288 |

**(a)** 该模型有多少可训练参数？以单精度浮点表示时，仅加载模型需要多少内存？  
**交付物**：一到两句话回答。

**(b)** GPT-2 XL 前向传播需要哪些矩阵乘法？总共需要多少 FLOPs（假设输入序列长度为 context_length）？  
**交付物**：矩阵乘法列表（含描述）及总 FLOPs。

**(c)** 模型哪些部分消耗最多 FLOPs？  
**交付物**：一到两句话回答。

**(d)** 对 GPT-2 small（12层，768维，12头）、medium（24层，1024维，16头）和 large（36层，1280维，20头）重复分析。随模型规模增大，各组件的 FLOPs 占比如何变化？  
**交付物**：各模型的组件 FLOPs 占比，及一到两句话描述。

**(e)** 将 GPT-2 XL 的上下文长度增至 16,384 时，总 FLOPs 如何变化？各组件的相对贡献如何变化？  
**交付物**：一到两句话回答。

---

## 4. 训练 Transformer LM

### 4.1 交叉熵损失

标准交叉熵（负对数似然）损失：

$$\ell(\theta; \mathcal{D}) = \frac{1}{|\mathcal{D}|m} \sum_{x \in \mathcal{D}} \sum_{i=1}^m -\log p_\theta(x_{i+1} | x_{1:i})$$

其中 Transformer 在每个位置 $i$ 输出 logits $o_i \in \mathbb{R}^{\text{vocab\_size}}$，通过 softmax 得到概率分布。

**困惑度**（Perplexity）用于评估时报告：

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^m \ell_i\right)$$

#### 问题（cross_entropy）：实现交叉熵（1分）

**交付物**：实现交叉熵损失函数，注意：
- 减去最大元素以保证数值稳定性
- 尽量消去 log 和 exp
- 处理额外批维度，返回批均值

实现适配器 `[adapters.run_cross_entropy]`，运行 `uv run pytest -k test_cross_entropy`。

---

### 4.2 SGD 优化器

**基础 SGD 更新**：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t)$$

实现时需继承 `torch.optim.Optimizer`，重写 `__init__` 和 `step` 方法。

#### 问题（learning_rate_tuning）：调整学习率（1分）

分别用学习率 `1e1`、`1e2`、`1e3` 运行 SGD 示例 10 步，观察每个学习率下的损失行为（更快减小、更慢减小还是发散）？  
**交付物**：一到两句话回答。

---

### 4.3 AdamW 优化器

AdamW 是对 Adam 优化器的改进，通过解耦权重衰减来改进正则化。

**AdamW 算法**（参数：学习率 $\alpha$，动量参数 $\beta_1, \beta_2$，权重衰减率 $\lambda$，数值稳定项 $\varepsilon$）：

1. $\alpha_t = \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
2. $\theta \leftarrow \theta - \alpha\lambda\theta$（权重衰减）
3. $m \leftarrow \beta_1 m + (1-\beta_1)g$
4. $v \leftarrow \beta_2 v + (1-\beta_2)g^2$
5. $\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v}+\varepsilon}$

典型超参数：$(\beta_1, \beta_2) = (0.9, 0.999)$，大型 LLM 常用 $(0.9, 0.95)$。

#### 问题（adamw）：实现 AdamW（2分）

**交付物**：实现 `AdamW` 为 `torch.optim.Optimizer` 的子类。

实现适配器 `[adapters.get_adamw_cls]`，运行 `uv run pytest -k test_adamw`。

#### 问题（adamw_accounting）：AdamW 训练资源核算（2分）

假设所有张量使用 `float32`：

**(a)** 运行 AdamW 的峰值内存需求是多少？分别列出参数、激活、梯度和优化器状态的内存，用超参数表示。  
**交付物**：各项的代数表达式及总量。

**(b)** 对 GPT-2 XL，最大能容纳多大的 batch size（80GB 显存限制）？  
**交付物**：形如 $a \cdot \text{batch\_size} + b$ 的表达式，及最大 batch size 的数值。

**(c)** 一步 AdamW 需要多少 FLOPs？  
**交付物**：代数表达式，附简短说明。

**(d)** 模型 FLOPs 利用率（MFU）定义为观测吞吐量与硬件理论峰值吞吐量之比。假设 MFU 为 50%，NVIDIA H100 理论峰值 495 TFLOP/s，batch size 为 1024，反向传播 FLOPs 为前向两倍，在单张 H100 上训练 GPT-2 XL 40万步需要多久？  
**交付物**：小时数及简短说明。

---

### 4.4 学习率调度

我们实现 LLaMA 使用的余弦退火调度：

- **预热阶段**：若 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{\max}$
- **余弦退火阶段**：若 $T_w \leq t \leq T_c$，则 $\alpha_t = \alpha_{\min} + \frac{1}{2}\left(1 + \cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max} - \alpha_{\min})$
- **退火后阶段**：若 $t > T_c$，则 $\alpha_t = \alpha_{\min}$

#### 问题（learning_rate_schedule）：实现余弦学习率调度（1分）

**交付物**：实现上述调度函数。

实现适配器 `[adapters.get_lr_cosine_schedule]`，运行 `uv run pytest -k test_get_lr_cosine_schedule`。

---

### 4.5 梯度裁剪

给定梯度 $g$，计算其 $\ell_2$ 范数 $\|g\|_2$。若 $\|g\|_2 < M$，保持不变；否则缩放为 $g \cdot \frac{M}{\|g\|_2 + \varepsilon}$（$\varepsilon = 10^{-6}$）。

#### 问题（gradient_clipping）：实现梯度裁剪（1分）

**交付物**：实现梯度裁剪函数，接受参数列表和最大 $\ell_2$ 范数，原地修改各参数梯度。

实现适配器 `[adapters.run_gradient_clipping]`，运行 `uv run pytest -k test_gradient_clipping`。

---

## 5. 训练循环

### 5.1 数据加载器

分词后的数据是单个词元序列 $x = (x_1, \ldots, x_n)$，数据加载器将其转换为批次流。每个批次包含 $B$ 条长度为 $m$ 的序列及对应的下一词元（目标）序列。

#### 问题（data_loading）：实现数据加载（2分）

**交付物**：实现一个函数，接受 numpy 数组、batch_size、context_length 和设备字符串，返回两个形状为 `(batch_size, context_length)` 的张量（输入和目标）。

实现适配器 `[adapters.run_get_batch]`，运行 `uv run pytest -k test_get_batch`。

> **内存映射提示**：使用 `np.memmap`（或 `np.load` 的 `mmap_mode='r'` 参数）以内存映射模式加载数据集，避免将大文件全部载入内存。

> **低资源提示**：CPU 用户使用 `'cpu'` 设备，Apple Silicon 用户使用 `'mps'` 设备。

---

### 5.2 检查点（Checkpointing）

检查点需保存所有恢复训练所需的状态：模型权重、优化器状态（如 AdamW 的动量估计）和迭代编号。

#### 问题（checkpointing）：实现模型检查点（1分）

**交付物**：实现两个函数：

```python
def save_checkpoint(model, optimizer, iteration, out)
    # 将模型、优化器和迭代次数的状态保存到文件

def load_checkpoint(src, model, optimizer)
    # 从文件加载检查点，恢复模型和优化器状态，返回迭代次数
```

实现适配器 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]`，运行 `uv run pytest -k test_checkpointing`。

---

### 5.3 训练循环

#### 问题（training_together）：整合（4分）

**交付物**：编写完整训练脚本，至少支持：
- 配置和控制各种模型和优化器超参数
- 使用 `np.memmap` 高效加载大型训练集和验证集
- 将检查点序列化到指定路径
- 定期记录训练和验证性能（如输出到控制台或 Weights and Biases）

---

## 6. 生成文本

### 解码过程

每步解码：给定序列 $x_{1\ldots t}$，通过以下方式采样下一词元：

$$P(x_{t+1} = i | x_{1\ldots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$$

$$v = \text{TransformerLM}(x_{1\ldots t})_t$$

重复采样（将生成的词元加入输入），直到生成 `<|endoftext|>` 或达到最大生成长度。

### 解码技巧

**温度缩放**（Temperature Scaling）：

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_j \exp(v_j/\tau)}$$

$\tau \to 0$ 时分布趋向独热向量（贪心解码）。

**核采样（Nucleus/Top-p Sampling）**：

$$P(x_{t+1} = i | q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{若 } i \in V(p) \\ 0 & \text{否则} \end{cases}$$

其中 $V(p)$ 是满足 $\sum_{j \in V(p)} q_j \geq p$ 的最小概率最高的词元集合。

#### 问题（decoding）：解码（3分）

**交付物**：实现语言模型解码函数，支持：
- 给定提示词生成补全（至 `<|endoftext|>` 或最大长度）
- 用户可控的最大生成词元数
- 温度缩放
- Top-p（核）采样

---

## 7. 实验

### 7.1 实验运行方式

为便于快速实验，我们将在小规模模型（约 1700万参数）和简单数据集（TinyStories）上进行。请确保定期评估验证损失，记录步数和墙钟时间，可使用 Weights and Biases 等日志工具。

#### 问题（experiment_log）：实验日志（3分）

**交付物**：实验追踪代码，及以下各问题的实验日志（记录所有尝试）和损失曲线。

---

### 7.2 TinyStories

**TinyStories 示例**：

> 从前，有一个小男孩叫 Ben。Ben 喜欢探索他周围的世界……（儿童故事风格文本）

#### 基础超参数

| 超参数 | 值 |
|--------|-----|
| vocab_size | 10,000 |
| context_length | 256 |
| d_model | 512 |
| d_ff | 1,344 |
| RoPE theta (Θ) | 10,000 |
| 层数 / 注意力头数 | 4层 / 16头 |
| 总处理词元数 | 327,680,000 |

需自行调优：学习率、学习率预热步数、AdamW 超参数（$\beta_1, \beta_2, \varepsilon$）和权重衰减。

#### 问题（learning_rate）：调整学习率（2 B200 GPU 小时）（3分）

**(a)** 对学习率进行超参数搜索，报告最终损失（或发散情况）。  
**交付物**：多个学习率对应的损失曲线；验证损失（每词元）≤ 1.45 的模型。

**(b)** 研究"稳定边缘"学习率与最优学习率的关系。  
**交付物**：包含至少一条发散曲线的损失对比图，及分析。

> **低资源提示**：CPU/MPS 用户将总词元数降至 40,000,000，验证损失目标放宽至 2.00。M4 Max 上 CPU 训练约 1 小时22分钟，MPS 约 36 分钟。

#### 问题（batch_size_experiment）：批次大小变化（1 B200 GPU 小时）（1分）

从 1 到 GPU 内存上限变化批次大小，至少包含 64 和 128 等典型值。  
**交付物**：不同批次大小的损失曲线，及几句关于批次大小对训练影响的分析。

#### 问题（generate）：生成文本（1分）

**交付物**：至少 256 词元（或至第一个 `<|endoftext|>`）的生成文本，对输出流畅度的简评，及至少两个影响输出质量的因素。

**参考输出示例**：

> 从前，有一个叫 Lily 的漂亮女孩。她喜欢嚼口香糖……（流畅的儿童故事）

---

### 7.3 消融实验与架构修改

#### 消融1：层归一化

**问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 GPU 小时）（1分）**

从 Transformer 中移除所有 RMSNorm 进行训练。用较低学习率能否恢复稳定性？  
**交付物**：移除 RMSNorm 后的学习曲线，及最佳学习率的曲线；几句关于 RMSNorm 影响的评述。

**问题（pre_norm_ablation）：实现 Post-norm 并训练（0.5 B200 GPU 小时）（1分）**

将 pre-norm 改为 post-norm：

$$z = \text{RMSNorm}(x + \text{MultiHeadSelfAttention}(x))$$
$$y = \text{RMSNorm}(z + \text{FFN}(z))$$

**交付物**：Post-norm 与 Pre-norm 的对比学习曲线。

#### 消融2：位置编码

**问题（no_pos_emb）：实现 NoPE（0.5 B200 GPU 小时）（1分）**

完全移除位置编码信息（NoPE），与 RoPE 进行实证对比。  
**交付物**：RoPE 与 NoPE 性能对比的学习曲线。

#### 消融3：SwiGLU vs. SiLU

**问题（swiglu_ablation）：SwiGLU vs. SiLU（0.5 B200 GPU 小时）（1分）**

对比 SwiGLU 与无门控的 SiLU 前馈网络（将 $d_{ff}$ 设为 $4 \times d_\text{model}$ 以匹配参数量）：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \text{SiLU}(W_1 x)$$

**交付物**：SwiGLU 与 SiLU 的对比学习曲线，及几句分析。

---

### 7.4 在 OpenWebText 上运行

**OpenWebText 示例**：

> 棒球前景杂志技术总监 Harry Pavlidis 在雇用 Jonathan Judge 时冒了一个险……（真实网络文本风格）

#### 问题（main_experiment）：OpenWebText 实验（2 B200 GPU 小时）（2分）

用相同模型架构和训练步数在 OpenWebText 上训练语言模型。  
**交付物**：OpenWebText 上的训练损失曲线；与 TinyStories 损失的对比分析；生成文本及质量评述（为什么即使相同计算预算下输出质量更差？）。

---

### 7.5 自定义改进 + 排行榜

#### 排行榜规则

- **时长**：在 B200 上最多运行 45 分钟
- **数据**：只能使用提供的 OpenWebText 训练集
- 其他方面自由发挥

参考资源：
- 最新开源 LLM（如 Llama 3、Qwen 2.5）
- NanoGPT 加速竞赛仓库（[github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)）

#### 问题（leaderboard）：排行榜（10 B200 GPU 小时）（6分）

**交付物**：最终验证损失，x 轴为墙钟时间（<45分钟）的学习曲线，及改进方案描述。验证损失需低于基准线 5.0。

提交地址：[github.com/stanford-cs336/assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)

---

## 参考文献

1. R. Eldan & Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" 2023.
2. A. Gokaslan et al., "OpenWebText corpus." 2019.
3. R. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units," ACL 2016.
4. C. Wang et al., "Neural Machine Translation with Byte-Level Subwords." 2019.
5. P. Gage, "A new algorithm for data compression," C Users Journal, 1994.
6. A. Radford et al., "Language Models are Unsupervised Multitask Learners." 2019.
7. A. Radford et al., "Improving Language Understanding by Generative Pre-Training." 2018.
8. A. Vaswani et al., "Attention is All you Need," NeurIPS 2017.
9. T. Q. Nguyen & J. Salazar, "Transformers without Tears," IWSLT 2019.
10. R. Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020.
11. J. L. Ba et al., "Layer Normalization." 2016.
12. H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.
13. B. Zhang & R. Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
14. A. Grattafiori et al., "The Llama 3 Herd of Models." 2024.
15. A. Yang et al., "Qwen2.5 Technical Report," 2024.
16. A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.
17. D. Hendrycks & K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with GELUs." 2016.
18. S. Elfwing et al., "Sigmoid-Weighted Linear Units for Neural Network Function Approximation." 2017.
19. Y. N. Dauphin et al., "Language Modeling with Gated Convolutional Networks." 2017.
20. N. Shazeer, "GLU Variants Improve Transformer." 2020.
21. J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
22. D. P. Kingma & J. Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
23. I. Loshchilov & F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019.
24. T. B. Brown et al., "Language Models are Few-Shot Learners," NeurIPS 2020.
25. J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.
26. J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.
27. A. Holtzman et al., "The Curious Case of Neural Text Degeneration," ICLR 2020.
28. Y.-H. H. Tsai et al., "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," EMNLP-IJCNLP 2019.
29. A. Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers," NeurIPS 2023.
