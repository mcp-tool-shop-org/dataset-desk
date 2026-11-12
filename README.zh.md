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

一个风格 LoRA 从其训练图像和描述中学习。以下两个明显的缺陷会破坏一切：
- **重复。** 同一张图片保存两次，或者两张渲染图相差几个像素，都会让模型重复学习同一张图片。
- **并非描述的描述。** 从生成提示中复制的描述，或者将同一描述粘贴到 54 张图片上，都无法向模型提供关于图片的任何信息。更糟糕的是，如果稍后将该文本用于测试搜索，它会衡量提示的召回率，而不是对图像的理解。

dataset-desk 会在项目文件夹中找到这两类问题，而无需更改文件夹。它会生成一份报告和一份待处理队列，供人员进行处理。**没有任何内容会被自动删除。**

## 它能做什么

| 命令 | 您将获得什么 |
|---|---|
| `scan` | 扫描项目到 `scan.jsonl` 和 `scan-report.json`，位于 `--out` 下，无需加载嵌入器 |
| `index` | 扫描项目（记录、波形元数据、PNG 生成数据），并将每个唯一的图像嵌入到 `--out` 下的缓存中 |
| `dupes` | 字节级完全相同的副本（sha256）、近重复项（考虑镜像关系的 256 位 dHash）和语义重复项（嵌入余弦值），以 JSON 格式呈现。它在一个 DINOv2 索引上使用经过校准的双信号规则。同一对象的两个视图永远不会配对。 |
| `search` | 与描述（`--text`）、图片（`--image`）或两者结合最接近的图像 |
| `coverage` | 所需主题列表中每个主题的最佳图像，以便显示空白 |
| `report` | 一份 Markdown 报告：重复项、内容和外观上最不典型的已批准图像、描述审核，以及每个描述是否找到了其对应的图像 |
| `sidecar-import` | 导入与具有相同前缀的 `.txt` 描述文件配对的图像文件夹（`--sidecars` 在 `index`/`report`/`compose` 上执行相同的操作），运行描述筛选器，报告缺少 `.txt` 文件的图像（不进行文本生成），并将所有内容写入 `--out` |
| `compose` | 将最终训练描述组合到 `--out` 中，使用每个项目的规则和导入的编辑（默认情况下，仅将标记的记录放入审核队列中，或者提供并排的草稿，并使用 `--all`，而不替换人工或文件中的文本），并编写lane-3 数据集清单（`{"items":[{"image","sha256","caption"}]}`），其中包含每个部分的来源 |
| `import-edits` | 将从 `description-worksheet.md` 导入的已编辑行重新导入到 `--out/seat-edits.jsonl` 中，并使用 `seat:<name>` 编写者 ID |
| `text-index` | 嵌入 `--source` 文件夹（或 rnd-data 检出）中的文本和代码文件，并使用模型自己的分词器计算块数 |
| `text-dupes` | 完全相同和近重复的文件。`--task code` 仅比较脚本。 |
| `text-search` | 与查询（`--text`）最接近的文件。`--task code` 使用模型的代码检索提示。 |
| `lineage` | 跨不同运行的相同脚本的版本，按相似度排序 |

**阈值** 来自工作室对其自身数据的校准。每个阈值都在代码中标记为临时值，并附带其来源的测量值：
- **图像：** 在 DINOv2 索引上，当 `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985` 时，一对图像被认为是重复的。其他索引使用 26 位内的 dHash，以及余弦值 ≥ 0.95。
- **文本：** 使用 bge-m3 时为 0.98，使用 EmbeddingGemma 2 时为 0.95。任何其他模型都需要 `--semantic`。

来自不同相机角度的同一物体的两个渲染图永远不会被认为是重复的。视图来自每个记录的 `provenance.asset_id` 和 `render_id`，或者来自文件名，例如 `<asset>__y<yaw>_e<elev>`。

**描述筛选器** 在没有任何模型的情况下运行。当描述满足以下条件时，它会被标记：
- 回显其提示：它等于提示或位于提示中，与提示共享 6 个或更多单词的连续序列，或者 90% 或更多（80% 或更多内容词）的单词出现在提示中；
- 包含记录的主题行：4 个或更多单词的 `canon.subject` 作为其单词的连续序列出现；
- 太短：少于 8 个单词或少于 5 个内容词；
- 复制了集合中的另一个描述：删除标点符号后完全相同，或者内容词的 Jaccard 系数为 0.85 或更高；
- 对于 SigLIP2 来说太长：超过 64 个令牌（唯一需要分词器的检查；未检查的记录永远不会被标记为通过）；
- 违反了每个项目的 `caption-rules.json` 规则（如果存在）：`prefix_missing`、`required_term_missing` 或 `banned_word`（以及池 `exact_duplicate` 和 `near_duplicate` 检查）。

标记的记录会发送到 `description-review.jsonl` 和一个**盲工作表**：每个标记的图像旁边都有 3-5 个在视觉上最接近的同类图像（如果缓存到 `--out` 下，则从 DINOv2 图像相似度中选择，否则从 dHash 中选择；绝不会使用文本嵌入，并且在锁定的池中）。没有提示、主题行和旧文本，因此重新编写的人会描述图片。可选的草稿编写器（默认情况下为 `--draft plain-sight`，或者 `--draft qwen3-vl`）可以仅从像素中编写第一个草稿，并将其放入审核队列中（绝不会直接放入训练描述中）；完整的原始草稿输出会保存到每个图像的 `drafter-raw.jsonl` 中。通过 `--all` 也会将未标记的记录作为并排的建议（`draft_offer` / `draft_offer_writer`）进行草稿编写，而不会替换人工或文件中的文本。经过人工编辑的草稿会被记录为 `captioner_assisted`。

**外观通道** 是每个图像的 30 个数字（在原始分辨率下的调色板、色调和纹理），因此“这张图片的渲染效果不好”与“这张图片的主题很奇怪”是分开显示的。

## 安装

```bash
pip install dataset-desk
```

这提供了**精简版本**：仅使用 numpy 和 Pillow，不使用 torch。重复检查、外观通道、描述筛选器和报告都可以在其中工作（`--backend none`）。

对于嵌入索引（搜索、覆盖率、语义重复、描述拟合检查），添加模型库：

```bash
pip install "dataset-desk[models]"
```

在配备 NVIDIA GPU 的 Windows 上，首先安装来自 `https://download.pytorch.org` 的 CUDA 构建版本的 torch；PyPI 的 Windows torch 仅为 CPU。

模型会在第一次使用时从 Hugging Face 处以固定的版本获取，然后从您的缓存中加载。每个任务都有一个默认值，并且 `--backend` 可以选择另一个：

| 后端 | 模型 | 默认值 |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | 文本 |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | 代码（`--task code`、`lineage`） |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

盲图像草稿编写器（`--draft`、`dataset_desk.registry.DRAFTER_REGISTRY`；两者都声明 `is_arm_model=False`，因此锁定的基准测试池允许它们进行草稿编写）：

| 草稿编写器 | 模型 | 固定版本 | 许可证 | 默认值 |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | 是 |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | — |

EmbeddingGemma 2 的使用方式与模型卡中描述的一致：每个命令使用一个任务前缀，仅加载文本用于文本处理，以及 `--dim 768|512|256`。将 `HF_HUB_OFFLINE=1` 设置为禁止下载。`plain-sight` 始终处于离线状态；`qwen3-vl` 遵守 `HF_HUB_OFFLINE`（`local_files_only=True`），拒绝任何版本不匹配（`MODEL_REVISION_MISMATCH`），遵守 VRAM 限制器，接受每个项目的提示（`drafter_prompt` 或 `--drafter-prompt`），并存储每个图像的提示和完整的原始输出。

**运行环境：**Python 3.11 和 3.12，在 Windows 和 Linux 上。默认使用 CPU，每个线程池都有上限（除非 `--threads` 另有说明，否则为 4 个线程），并且模型运行会将进程优先级降低到低于正常水平，以便它可以共享一台机器。

## 使用方法

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

# Training captions: audit + blind draft -> seat edit -> re-import -> compose manifest under --out
dataset-desk index        --project path/to/project --backend dinov2 --out desk-out/project
dataset-desk report       --project path/to/project --backend dinov2 --out desk-out/project --draft qwen3-vl --device cuda --grant g1
dataset-desk import-edits --project path/to/project --out desk-out/project --seat studio-1
dataset-desk compose      --project path/to/project --out desk-out/project --draft qwen3-vl --device cuda --grant g1

# Training folders with image + same-stem .txt sidecars (no records/ directory)
dataset-desk sidecar-import --project path/to/training-copy --out desk-out/training-copy
dataset-desk index          --project path/to/training-copy --out desk-out/training-copy --backend none --sidecars
dataset-desk report         --project path/to/training-copy --out desk-out/training-copy --backend none
```

### 每个项目的标题规则（`caption-rules.json`）和来源

将 `caption-rules.json` 放在 `<project>/caption-rules.json`（或传递 `--caption-config path/to/rules.json`）：

```json
{
  "multimodal_pool": "salt-road",
  "prefix": "linocut print,",
  "drafter": "qwen3-vl",
  "drafter_prompt": "Describe only the visible pose, props, backdrop, and floor in 12 to 25 words.",
  "class_field": "canon.category",
  "class_pattern": "(?:^|/)(?P<class>[a-z0-9_-]+)_[0-9]+\\.[a-z0-9]+$",
  "backdrop": {
    "text": "plain pale ground",
    "field": "canon.backdrop",
    "required": false,
    "banned_words": ["neon", "bokeh"]
  },
  "banned_words": ["masterpiece", "photorealistic"],
  "classes": {
    "vessel": {
      "required_terms": ["carved hull", "rigging lines"],
      "banned_words": ["portrait", "person"]
    }
  }
}
```

- **门控检查（与模型无关，报告在 `description-review.jsonl` 中）：** `prefix_missing`（标题不以 `prefix` 开头），`required_term_missing`（记录的类别中缺少任何 `required_terms` 条目——从 `class_field` 或 `class_pattern` 解析得出——或者缺少 `required` 背景），`banned_word`（任何全局、背景或类别 `banned_words` 短语出现），以及池 `exact_duplicate` 和 `near_duplicate`（Jaccard ≥ 0.85）。
- **组合（`compose` 或 `report --compose`）：** 在 `style-dataset-lab` 的第 3 栏中写入 `<out>/captions-manifest.json`（`--manifest-name` 覆盖文件名），格式为 `run_dataset` `{"items":[{"image","sha256","caption",...}]}`——绝不在 `--project` 内部。每个标题都由 `prefix`、逐字 `class` 必需的术语、可选的 `backdrop` 以及自由可见内容部分（`seat` 导入时进行编辑，否则使用已批准的人工/辅助文本——绝不使用未经编辑的草稿）组合而成。默认情况下，`compose` 和 `report --compose` 草稿**仅**将模型无关的门控标记记录到审核队列中；`--all` 还会将可选的 `draft_offer` / `draft_offer_writer` 建议与未标记的原始记录一起写入，而不会替换人工或辅助文本。
- **来源记录：** `captions-manifest.json` 和 `description-review.jsonl` 中的每个最终标题都包含一个逐部分 `writers` 列表，使用 `config:<file>:<line>`、`drafter:<id>@<rev>:<prompt sha>`、`seat:<name>`、`record:<file>` 和 `sidecar:<file>`，并在由座席在查看草稿或建议后编辑时使用图层 `captioner_assisted`，对于人工记录使用 `"human"`，对于 `.txt` 辅助文件使用 `"sidecar"`。
- **辅助文件导入（`sidecar-import` 或 `--sidecars`）：** 将每个图像与其具有相同前缀的 `.txt` 辅助文件配对，分配 `author_kind: "sidecar"`（绝不分配 `"human"`），提取嵌入的 ComfyUI PNG 生成提示（如果存在），运行完整的描述门控（提示回显、精确/Jaccard 重复以及 `caption-rules.json`），报告任何缺少 `.txt` 辅助文件的图像，而不是发明文本，并将所有导入的记录和审核文件严格地写入 `--out`。
- **按标签锁定池、失败时重命名副本保护以及 `pool_detection` 状态：** 当 `realpath` 文件夹名称或 `"multimodal_pool"` 在配置中（`caption-rules.json` 或 `project.json`）命名一个锁定的池时，`check_multimodal_pool` 会强制执行基准测试锁定，涵盖 `--project` 和文本命令（`text-index`、`text-dupes`、`text-search`、`lineage`）。对于未标记的文件夹（不在这些名称之外），它会针对非空本地清单（`DATASET_DESK_POOL_MANIFEST` 或 `pool_manifest` 在 `~/.config/dataset-desk/config.json` 中，保存在仓库之外）或 `--locked-pool-dir <path>`，对扫描读取的图像的精确集合（包括符号链接和 Windows 目录连接）进行哈希处理——并且对于文本命令，还会检查 `.txt` 标题辅助文件前缀与锁定的池图像前缀是否匹配——如果匹配，则拒绝并以失败方式（`PERM_LOCKED_POOL`）命名匹配的池和计数。每个命令都会解析一个 `pool_detection` 状态（`status: "on"`，每个已配置的池和每个来源的计数至少有一个图像哈希值——绝不使用哈希值——或者 `status: "off"`，并发出警告文本，一次性输出到 `stderr`），并在 `<out>/pool-detection.json` 中记录 `pool_detection`，JSON/JSONL 输出位于 `--out` 下，并且在每个 Markdown 报告中添加一行注释。
- 重命名副本检测通过 `sha256` 匹配图像，因此，仅由新图像（没有与锁定的池共享）制作的副本，或者经过调整大小、重新编码或删除元数据的副本，都不会通过 `sha256` 匹配；使用 `"multimodal_pool"` 在 `caption-rules.json` 或 `project.json` 中标记此类项目。

`dataset-desk --help` 列出了每个标志。退出代码：`0` 正常，`1` 用户错误或拒绝，`2` 意外错误。错误以 `error <CODE>: <message>` 的形式打印，并带有 `hint:` 行；`--debug` 添加堆栈跟踪。

**它读取的项目布局** 是 [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) 的：

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

即使没有 `records/` 的项目也可以正常工作，只要图像位于 `outputs/` 下方的文件夹中（跳过位于 `outputs/` 顶部的文件，因为它们是联系表）：每个图像都是“未记录的”，并且仍然会运行重复和外观检查。

## 威胁模型和信任

- **它读取的内容：** 您使用 `--project` 命名的项目文件夹：图像、`records/*.json`、波形元数据以及嵌入在 PNG 文件中的生成数据。文本命令读取您命名的 `--source` 文件夹或 `--rnd-data` 检出。它们拒绝 rnd-data 的私有辅助文件和 blob 存储，并且仅打印相对路径。
- **它写入的内容：** 仅在 `--out` 下：其扫描缓存、嵌入缓存、报告、`description-review.jsonl` 和工作表。它拒绝项目内部的 `--out`（`PERM_OUT_INSIDE_PROJECT`），因此项目永远不会被修改。
- **它永远不会触及的内容：** 两个文件夹之外的任何内容。它不会删除任何内容，也不会在任何地方删除。
- **网络：** 精简版本永远不会使用它。模型后端在首次使用时从 Hugging Face 获取其固定的权重，除非设置了 `HF_HUB_OFFLINE=1`；草稿加载器每次都使用 `HF_HUB_OFFLINE=1` 和 `local_files_only=True`。没有模型会运行远程代码。
- **权限：** 一个普通的用户帐户。GPU 需要 `--device` 加上一个 `--grant` ID，因此不会意外地占用共享显卡。在加载模型之前，VRAM 限制器会检查已使用的内存加上模型的测量需求是否低于 `--vram-budget`（默认值为 26,624 MiB）。它永远不会驱逐任何内容；`--wait` 会轮询，直到负载适合为止。
- **秘密和遥测：** 它不读取任何凭据，并且不收集或发送任何遥测数据。
- **受保护的步骤：** 对于锁定的运行基准测试的项目，除非传递了 `--allow-multimodal-pool`，否则拒绝描述嵌入和任何基于模型的描述接受，因此基准测试的查询不会受到它所测量的模型的影响。

如 [SECURITY.md](SECURITY.md) 中所述，报告漏洞。

## 状态

**v0.3.0，一个公共原型。** 代码在离线环境下进行测试，使用小型模拟数据代替模型库。
- **校准：** 该工作室使用自己的数据在 RTX 5090 上运行，设置重复阈值，这些阈值是暂定的，并且基于每种类型的一个数据池。VRAM 数据是保守的整卡数据，包括 CUDA 上下文，在 RTX 5090 上测量；其他显卡和数据集可能会有所不同。代码阈值基于 17 个已植入的副本。该方法和数据可在[新闻简报第 16 期](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/)中找到。
- **大型项目：** 重复项、异常值和外观通道在有限的块中工作。在合成数据集上，近重复搜索和外观通道在 10 万张图像上只需几分钟，而在以前则需要数小时或耗尽内存。描述门控的近重复检查使用精确的前缀过滤器，因此它仅比较可能达到 0.85 的描述。第一次扫描每张图像的速度大约快 2 倍，结果相同，并且在 `--threads` 下并行运行。
- **预计在 1.0 版本之前，命令行会发生变化。**

设计记录、其研究依据和审查历史记录都保存在工作室的研究库中：[rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)。

## 开发

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI 保持代码覆盖率在 90% 或更高。

## 许可证

[MIT](LICENSE)。它可以加载的模型不属于此软件包，并且每个模型都有自己的许可证；在使用模型之前，请阅读模型卡。

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
