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

一种风格 LoRA 从其训练图像及其标题中学习。以下两个明显的缺陷会破坏一切：
- **重复。** 同一张图片保存两次，或者两张渲染图相差几个像素，都会让模型重复学习同一张图片。
- **并非描述的描述。** 从生成提示中复制的标题，或者将同一条规则粘贴到 54 张图片上，都无法向模型提供关于图片的任何信息。更糟糕的是，如果稍后将这些文本用于测试搜索，它会衡量提示的召回率，而不是对图像的理解。

dataset-desk 会在项目文件夹中找到这些问题，而无需更改文件夹。它会生成一份报告和一份待处理队列，供人员采取行动。**没有任何内容会被自动删除。**

## 它能做什么

| 命令 | 你将获得什么 |
|---|---|
| `index` | 扫描项目（记录、波形元数据、PNG 生成数据），并将每个唯一的图像嵌入到 `--out` 下的缓存中。 |
| `dupes` | 字节级完全相同的副本（sha256）、近重复项（256 位 dHash，差异在 10 位以内）和语义重复项（嵌入余弦值），以 JSON 格式呈现。 |
| `search` | 与描述（`--text`）、图片（`--image`）或两者结合最近似的图像。 |
| `coverage` | 所需主题列表中每个主题的最佳图像，以便显示空白。 |
| `report` | 一份 Markdown 报告：重复项、内容和外观上最不典型的已批准图像、描述审核，以及每个描述是否找到了其对应的图像。 |

**描述筛选器**无需任何模型即可运行。当描述满足以下条件时，该描述会被标记：
- 回显其提示：它与提示相同或位于提示内部，与提示共享 6 个或更多单词的连续序列，或者 90% 或更多（80% 或更多内容词）的单词出现在提示中；
- 包含记录的主题行：4 个或更多单词的 `canon.subject` 作为其单词的连续序列出现；
- 太短：少于 8 个单词或少于 5 个内容词；
- 复制了集合中的另一个描述：删除标点符号后完全相同，或者内容词的 Jaccard 系数为 0.85 或更高；
- 对于 SigLIP2 来说太长：超过 64 个令牌（这是唯一需要分词器的检查；未检查的记录绝不会被标记为通过）。

标记的记录会发送到 `description-review.jsonl` 和一个**盲工作表**：每个标记的图像旁边都有 3–5 个在视觉上最相似的图像，不显示任何提示、主题行或旧文本，以便重新编写的人可以描述该图像。一个可选的草稿编写器（[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)，Florence-2）可以仅根据像素编写第一个草稿；草稿会记录为 `captioner_assisted`，然后由人员进行编辑。

**外观通道**是每个图像的 30 个数字（在原始分辨率下的调色板、色调和纹理），因此“这张图像的渲染效果不佳”与“这张图像的主题很奇怪”会分别显示。

## 安装

```bash
pip install "dataset-desk @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

这提供了**精简版本**：仅使用 numpy 和 Pillow，不使用 torch。重复检查、外观通道、描述筛选器和报告都可以在其中工作（`--backend none`）。

对于嵌入索引（搜索、覆盖率、语义重复项、描述拟合检查），添加模型库：

```bash
pip install "dataset-desk[models] @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

模型会在第一次使用时从 Hugging Face 获取固定版本的权重，然后从你的缓存中加载：[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)（`--backend eg2`）或 [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384)（`--backend siglip2`）。将 `HF_HUB_OFFLINE=1` 设置为禁止下载。plain-sight 草稿编写器始终处于离线状态：它绝不会下载。

**可在以下系统上运行：**Python 3.11 和 3.12，Windows 和 Linux。默认情况下使用 CPU，并且每个线程池都受到限制（除非 `--threads` 另有说明，否则为 4 个线程），并且模型运行会将进程优先级降低到低于正常优先级，以便它可以共享一台机器。

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
```

`dataset-desk --help` 列出了每个标记。退出代码：`0` 正常，`1` 用户错误或拒绝，`2` 意外错误。错误会以 `error <CODE>: <message>` 形式打印，并带有 `hint:` 行；`--debug` 会添加堆栈跟踪。

**它读取的项目布局**是 [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) 的：

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

即使没有 `records/` 的项目也可以工作，只要图像位于 `outputs/` 下的文件夹中（位于 `outputs/` 顶部的文件将被跳过，因为它们是联系表）：每张图像都被视为“未记录”，并且仍然会运行重复和外观检查。

## 威胁模型和信任

- **它读取的内容：**你使用 `--project` 命名的项目文件夹：图像、`records/*.json`、波形元数据以及嵌入到 PNG 文件中的生成数据。
- **它写入的内容：**仅在 `--out` 下：其扫描缓存、嵌入缓存、报告、`description-review.jsonl` 和工作表。它会拒绝项目内部的 `--out`（`PERM_OUT_INSIDE_PROJECT`），因此项目绝不会被修改。
- **它绝不会触及的内容：**这两个文件夹之外的任何内容。它不会在任何地方删除任何内容。
- **网络：**精简版本绝不会使用它。模型后端会在第一次使用时从 Hugging Face 获取其固定权重，除非设置了 `HF_HUB_OFFLINE=1`；草稿编写器每次都会使用 `HF_HUB_OFFLINE=1` 和 `local_files_only=True` 加载。没有模型会运行远程代码。
- **权限：**一个普通的用户帐户。GPU 需要 `--device` 加上一个 `--grant` ID，因此不会意外占用共享显卡。
- **秘密和遥测：**它不会读取任何凭据，并且不会收集或发送任何遥测数据。
- **受保护的步骤：**对于已锁定以进行运行基准测试的项目，除非传递了 `--allow-multimodal-pool`，否则描述嵌入和任何基于模型的描述接受都会被拒绝，因此基准测试的查询不会受到它所测量的模型的影响。

如 [SECURITY.md](SECURITY.md) 中所述，报告漏洞。

## 状态

**v0.1.2，一个公共原型。**该代码已在离线环境中进行了测试（152 个测试，96% 的行覆盖率，并使用小型替代模型库），但尚未在真实数据上进行校准：近重复项和语义阈值是起点。预计在 1.0 之前，命令行会发生变化。

设计记录、研究依据以及审查历史都保存在工作室的研究库中：[rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)。

## 开发

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

持续集成（CI）确保代码覆盖率达到 90% 或更高。

## 许可

[MIT](LICENSE)。它可以加载的模型不属于此软件包，并且每个模型都有自己的许可协议；在使用模型之前，请阅读模型说明。

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
