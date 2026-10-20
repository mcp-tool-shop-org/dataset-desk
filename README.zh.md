<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/dataset-desk/readme.png" alt="dataset-desk" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk"><img src="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/dataset-desk/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

<p align="center">
  A desk for making and reviewing visual datasets: find the copies, the near-copies and the odd ones out,<br>
  search a set by words or by picture, and check that every image's description really describes it.
</p>

---

## 为什么

风格 LoRA 从其训练图像及其描述中学习。以下两个明显的缺陷会破坏一切：
- **重复。** 同一张图片被保存两次，或者两张渲染图相差几个像素，都会让模型重复学习同一张图像。
- **并非描述的描述。** 从生成提示中复制的描述，或者将同一描述粘贴到 54 张图像上，都无法向模型提供关于图像的任何信息。更糟糕的是，如果稍后将该文本用于测试搜索，它会衡量提示的召回率，而不是对图像的理解。

dataset-desk 会在项目文件夹中找到这两类问题，而无需更改文件夹。它会生成一份报告和一份待处理队列，供人员进行处理。**不会自动删除任何内容。**

## 它所做的事情

| 命令 | 您将获得的结果 |
|---|---|
| `index` | 扫描项目（记录、波形元数据、PNG 生成数据），并将每个唯一的图像嵌入到 `--out` 下的缓存中。 |
| `dupes` | 字节级完全相同的副本（sha256）、近重复项（考虑镜像关系的 256 位 dHash）和语义重复项（嵌入余弦值），以 JSON 格式呈现。在 DINOv2 索引上，它使用经过校准的双信号规则。同一对象的两个视图永远不会配对。 |
| `search` | 与描述（`--text`）、图像（`--image`）或两者结合最近似的图像 |
| `coverage` | 所需主题列表中每个主题的最佳图像，以便显示空白 |
| `report` | 一份 Markdown 报告：重复项、内容和外观上最不典型的已批准图像、描述审核，以及每个描述是否找到了其对应的图像。 |
| `text-index` | 嵌入 `--source` 文件夹（或 rnd-data 检出）中的文本和代码文件，并使用模型自身的 tokenizer 统计块的数量。 |
| `text-dupes` | 完全相同和近重复的文件。`--task code` 仅比较脚本。 |
| `text-search` | 与查询（`--text`）最接近的文件。`--task code` 使用模型的代码检索提示。 |
| `lineage` | 跨不同运行的同一脚本的版本，按相似度排序。 |

**阈值** 来自工作室对其自身数据的校准。代码中标记了每个阈值，并附带了其来源的测量值：
- **图像：** 在 DINOv2 索引上，当 `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985` 时，一对图像被认为是重复的。其他索引使用 26 位内的 dHash，以及余弦值 ≥ 0.95。
- **文本：** 使用 bge-m3 时为 0.98，使用 EmbeddingGemma 2 时为 0.95。任何其他模型都需要 `--semantic`。

来自不同相机角度的同一物体的两个渲染图永远不会被认为是重复的。视图来自每个记录的 `provenance.asset_id` 和 `render_id`，或者来自文件名（如 `<asset>__y<yaw>_e<elev>`）。

**描述门控** 在没有任何模型的情况下运行。当描述满足以下条件时，会被标记：
- 回显其提示：它等于提示或位于提示中，与提示共享 6 个或更多单词的连续序列，或者 90% 或更多（80% 或更多内容词）的单词出现在提示中；
- 包含记录的主题行：4 个或更多单词的 `canon.subject` 作为其单词的连续序列出现；
- 太短：少于 8 个单词或少于 5 个内容词；
- 复制了集合中的另一个描述：删除标点符号后完全相同，或者内容词的 Jaccard 系数为 0.85 或更高；
- 对于 SigLIP2 来说太长：超过 64 个令牌（这是唯一需要 tokenizer 的检查；未检查的记录永远不会被标记为通过）。

被标记的记录会发送到 `description-review.jsonl` 和一个**盲目工作表**：每个被标记的图像旁边都有 3-5 个在视觉上最接近的图像，不显示提示、主题行或旧文本，以便重新编写的人可以描述图像。一个可选的草稿编写器（[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)，Florence-2）可以仅从像素中编写第一个草稿；草稿会记录为 `captioner_assisted`，然后由人员进行编辑。

**外观通道** 是每个图像的 30 个数字（在原始分辨率下的调色板、色调和纹理），因此“这张图像的渲染效果不对”与“这张图像的主题很奇怪”会分别显示。

## 安装

```bash
pip install dataset-desk
```

这提供了**精简版本**：仅使用 numpy 和 Pillow，不使用 torch。重复检查、外观通道、描述门控和报告都可以在其中工作（`--backend none`）。

对于嵌入索引（搜索、覆盖率、语义重复项、描述拟合检查），添加模型库：

```bash
pip install "dataset-desk[models]"
```

模型会在第一次使用时从 Hugging Face 获取固定版本的模型，然后从您的缓存中加载。每个任务都有一个默认值，并且 `--backend` 可以选择另一个：

| 后端 | 模型 | 默认用于 |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | 文本 |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | 代码（`--task code`，`lineage`） |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

EmbeddingGemma 2 的使用方式与模型卡中描述的方式相同：每个命令使用任务前缀，仅加载文本用于文本，以及 `--dim 768|512|256`。将 `HF_HUB_OFFLINE=1` 设置为禁止下载。plain-sight 草稿编写器始终处于离线状态：它永远不会下载。

**运行于：** Python 3.11 和 3.12，在 Windows 和 Linux 上。默认情况下使用 CPU，并且每个线程池都受到限制（除非 `--threads` 另有说明，否则为 4 个线程），并且模型运行会将进程的优先级降低到低于正常优先级，以便它可以共享一台机器。

## 用法

```bash
# The lite pass: no model, no GPU
dataset-desk index  --project path/to/project --backend none --out desk-out/project
dataset-desk report --project path/to/project --backend none --out desk-out/project

# With an embedding index
dataset-desk index    --project path/to/project --backend siglip2 --out desk-out/project
dataset-desk search   --project path/to/project --backend siglip2 --out desk-out/project --text "a squat bog creature"
dataset-desk coverage --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt
dataset-desk report   --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt

# Image duplicates with the calibrated DINOv2 rule
dataset-desk index --project path/to/project --backend dinov2 --out desk-out/project
dataset-desk dupes --project path/to/project --out desk-out/project

# Text and code files
dataset-desk text-index  --source path/to/folder --out desk-out/texts               # bge-m3, for text
dataset-desk text-dupes  --source path/to/folder --out desk-out/texts
dataset-desk text-index  --source path/to/folder --out desk-out/texts --task code   # EmbeddingGemma 2, for code
dataset-desk text-search --source path/to/folder --out desk-out/texts --task code --text "where the cache is written"
```

`dataset-desk --help` 列出了每个标志。退出代码：`0` 正常，`1` 用户错误或拒绝，`2` 意外错误。错误会以 `error <CODE>: <message>` 形式打印，并带有 `hint:` 行；`--debug` 会添加堆栈跟踪。

**它读取的项目布局** 是 [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) 的：

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

即使没有 `records/` 的项目也可以工作，只要图像位于 `outputs/` 下的文件夹中（位于 `outputs/` 顶部的文件将被跳过，作为联系表）：每张图像都被视为“未记录”，并且仍然会运行重复和外观检查。

## 威胁模型和信任

- **它读取的内容：**项目文件夹，您将其命名为 `--project`：图像、`records/*.json`、波形元数据以及嵌入在 PNG 文件中的生成数据。文本命令读取您命名的 `--source` 文件夹或 `--rnd-data` 检出版本。它拒绝使用 rnd-data 的私有辅助文件和 blob 存储，并且它仅打印相对路径。
- **它写入的内容：**仅在 `--out` 下：它的扫描缓存、嵌入缓存、报告、`description-review.jsonl` 和工作表。它拒绝在项目（`PERM_OUT_INSIDE_PROJECT`）内部使用 `--out`，因此项目永远不会被修改。
- **它永远不会触及的内容：**这两个文件夹之外的任何内容。它不会删除任何内容，也不会在任何地方删除。
- **网络：**精简版本绝不会使用它。模型后端在首次使用时，会从 Hugging Face 获取其已固定的权重，除非设置了 `HF_HUB_OFFLINE=1`；绘图器每次加载时都使用 `HF_HUB_OFFLINE=1` 和 `local_files_only=True`。没有模型会运行远程代码。
- **权限：**一个普通的用户帐户。GPU 需要 `--device` 加上一个 `--grant` ID，因此不会意外地占用共享显卡。在加载模型之前，VRAM 监管器会检查已使用的内存加上模型的测量需求是否符合 `--vram-budget`（默认值为 26,624 MiB）。它永远不会驱逐任何内容；`--wait` 会轮询，直到加载完成。
- **密钥和遥测数据：**它不读取任何凭据，并且不收集或发送任何遥测数据。
- **受保护的步骤：**对于已锁定以进行运行基准测试的项目，除非传递了 `--allow-multimodal-pool`，否则拒绝使用描述嵌入和任何基于模型的描述接受，因此基准测试的查询不会受到它所测量的模型的影响。

如 [SECURITY.md](SECURITY.md) 中所述，请报告漏洞。

## 状态

**v0.2.0，一个公共原型。**代码已在离线状态下进行了测试：188 个测试和 95% 的代码覆盖率，并使用小型替代方案来代替模型库。
- **校准：**工作室在其自身的数据上，在一块 RTX 5090 上运行，以设置重复阈值和 VRAM 值。这些值是暂定的，并且基于每种类型的一个池。代码阈值基于 17 个已植入的副本。该方法和数据位于 [newsletter issue 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/) 中。
- **大型项目：**重复项、异常值和查找通道在有界块中工作。在合成数据集上，近重复搜索和查找通道在 100,000 张图像上花费几分钟，而在以前则需要数小时或耗尽内存。描述门大约快 6 倍。它的 Jaccard 检查在共享许多单词的非常大的数据集上仍然会显著减慢，并且这个问题尚未解决。
- **预计在 1.0 之前命令行会发生变化。**

设计记录、其研究基础和其审查历史记录都位于工作室的研究库中：[rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)。

## 开发

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI 将代码覆盖率保持在 90% 或更高。

## 许可证

[MIT](LICENSE)。它可以加载的模型不属于此软件包，并且每个模型都具有自己的许可证；在使用模型之前，请阅读模型卡。

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
