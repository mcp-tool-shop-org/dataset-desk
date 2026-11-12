<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## なぜか

スタイルLoRAは、トレーニング画像とそのキャプションから学習します。以下の2つの問題が両方を台無しにします。
- **繰り返し:** 同じ画像を2回保存したり、数ピクセル異なる2つの画像をレンダリングしたりすると、1つの画像が2回学習されます。
- **説明ではない説明:** 生成プロンプトからコピーされたキャプション、または54枚の画像に貼り付けられたルールは、モデルに画像に関する情報を何も伝えません。さらに悪いことに、そのテキストが後で検索のテストに使用された場合、それは画像の理解ではなく、プロンプトの再現率を測定します。

dataset-deskは、プロジェクトフォルダー上でこれら両方を変更せずに検出します。レポートとレビューキューを作成し、担当者が対応できるようにします。**自動的に削除されるものは何もありません。**

## 機能

| コマンド | 結果 |
|---|---|
| `scan` | エンベッダーをロードせずに、プロジェクトを`scan.jsonl`と`scan-report.json`の下の`--out`にスキャンします。 |
| `index` | プロジェクト（レコード、波形メタデータ、PNG生成データ）をスキャンし、すべてのユニークな画像を`--out`の下のキャッシュにエンベ​​ドします。 |
| `dupes` | バイト単位で同一のコピー（sha256）、ほぼ重複するコピー（ミラーを考慮した256ビットのdHash）、および意味的に重複するコピー（エンベディングのコサイン類似度）をJSON形式で出力します。DINOv2インデックスでは、較正された二信号ルールを使用します。一つのオブジェクトの二つのビューは決してペアリングされません。 |
| `search` | 説明（`--text`）、画像（`--image`）、またはその両方に対して最も近い画像 |
| `coverage` | 必要な被写体リスト内の各被写体に対して最適な画像を表示し、ギャップを明らかにします。 |
| `report` | Markdown形式のレポート：重複、コンテンツと見た目に基づいて最も一般的でない承認された画像、説明の監査、および各説明が独自の画像を見つけられるかどうか。 |
| `sidecar-import` | 画像と同じ語幹の`.txt`キャプションサイドカーがペアになったフォルダをインポートし（`index`/`report`/`compose`の`--sidecars`でも同じ処理を行います）、説明ゲートを実行し、`.txt`サイドカーのない画像はテキストを作り出さずに報告し、すべてを`--out`の下に書き出します |
| `compose` | プロジェクトごとのルールとインポートされたシート編集から、最終的なトレーニングキャプションを`--out`にまとめます（デフォルトでは、フラグが立てられたレコードのみをレビューキューにドラフトするか、または`--all`を使用して並べてドラフトを表示し、人間またはサイドカーテキストを置き換えない）。パートごとの出所情報を含む、レイヤー3のデータセットマニフェスト（`{"items":[{"image","sha256","caption"}]}`）を作成します。 |
| `import-edits` | 編集された行を`description-worksheet.md`から`--out/seat-edits.jsonl`に`seat:<name>`の書き込みIDで再インポートします。 |
| `text-index` | `--source`フォルダー（またはrnd-dataチェックアウト）のテキストファイルとコードファイルをエンベ​​ドし、モデル自身のトークナイザーを使用してチャンクをカウントします。 |
| `text-dupes` | 正確に一致するファイルと、ほぼ重複するファイル。`--task code`はスクリプトのみを比較します。 |
| `text-search` | クエリ（`--text`）に対して最も近いファイル。`--task code`は、モデルのコード検索プロンプトを使用します。 |
| `lineage` | 実行間で類似度に基づいてランク付けされた、同じスクリプトのバージョン。 |

**閾値**は、スタジオが独自のデータで実行したキャリブレーションから得られます。それぞれがコード内で一時的なものとしてマークされ、測定値が記録されます。
- **画像:** DINOv2インデックスでは、ペアが`(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`の場合に重複と見なされます。他のインデックスでは、26ビット以内のdHashと、コサイン類似度≥0.95を使用します。
- **テキスト:** bge-m3を使用すると0.98、コードの場合はEmbeddingGemma 2を使用すると0.95です。他のモデルでは、`--semantic`が必要です。

異なるカメラアングルから見た同じオブジェクトの2つのレンダリングは、決して重複しません。ビューは、各レコードの`provenance.asset_id`と`render_id`、またはファイル名（例：`<asset>__y<yaw>_e<elev>`）から取得されます。

**説明ゲート**は、モデルを使用せずに実行されます。説明がフラグが立てられるのは、次の場合です。
- プロンプトをエコーする：プロンプトと等しいか、プロンプト内に含まれているか、6語以上の連続した単語を共有しているか、または90％以上の単語（コンテンツ単語の80％以上）がプロンプトに含まれている。
- レコードの件名行をラップする：4語以上の`canon.subject`が、その単語の連続したシーケンスとして表示される。
- 短すぎる：8語未満または5語未満のコンテンツ単語。
- セット内の別の説明と重複する：句読点を削除すると同一になるか、コンテンツ単語のJaccard係数が0.85以上。
- SigLIP2に対して長すぎる：64トークン以上（トークナイザーが必要な唯一のチェック。チェックされていないレコードは、合格としてマークされることはありません）。
- プロジェクトごとの`caption-rules.json`ルール（存在する場合）に違反する：`prefix_missing`、`required_term_missing`、または`banned_word`（プール内の`exact_duplicate`および`near_duplicate`チェックに加えて）。

フラグが立てられたレコードは、`description-review.jsonl`と「非表示のワークシート」に送られます。各フラグが立てられた画像には、視覚的に最も近い3～5個の画像（`--out`の下にキャッシュされている場合、DINOv2の画像のみの類似度から選択されます。そうでない場合は、dHash。ロックされたプール内のテキスト埋め込みは使用されません）が並べられ、プロンプト、件名、または以前のテキストは表示されません。そのため、書き直す人は画像の内容を説明します。 オプションのドラフター（デフォルトは`--draft plain-sight`、または`--draft qwen3-vl`）を使用すると、画像のピクセル情報のみから最初のドラフトを作成し、それをレビューキューに送信できます（トレーニングキャプションに直接送信されることはありません）。完全な生のドラフター出力は、画像ごとに`drafter-raw.jsonl`に保存されます。 `--all`を通過させることで、フラグが立てられていないレコードも、人間の作成したテキストや補助テキストを置き換えることなく、並べて表示される提案（`draft_offer` / `draft_offer_writer`）としてドラフトされます。 座席編集後のドラフトは、`captioner_assisted`として記録されます。

**ルックチャネル**は、画像ごとに30個の数値（ネイティブ解像度でのパレット、トーン、およびテクスチャ）です。そのため、「この画像のレンダリングがおかしい」という問題は、「この画像の被写体が奇妙だ」という問題とは別に表示されます。

## インストール

```bash
pip install dataset-desk
```

これにより、**ライトティア**が提供されます。numpyとPillowのみを使用し、torchは使用しません。重複チェック、ルックチャネル、説明ゲート、およびレポートはすべて、この環境で機能します（`--backend none`）。

エンベディングインデックス（検索、カバレッジ、意味的な重複、説明の適合性チェック）の場合は、モデルライブラリを追加します。

```bash
pip install "dataset-desk[models]"
```

WindowsでNVIDIA GPUを使用する場合は、まず`https://download.pytorch.org`からtorchのCUDAビルドをインストールします。PyPIのWindows torchはCPU専用です。

モデルは、最初に使用されたときにHugging Faceから固定されたリビジョンで取得され、その後はキャッシュからロードされます。各タスクにはデフォルトがあり、`--backend`が別のモデルを選択します。

| バックエンド | モデル | デフォルト |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | テキスト |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | コード（`--task code`、`lineage`） |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

ブラインド画像ドラフター（`--draft`、`dataset_desk.registry.DRAFTER_REGISTRY`。どちらも`is_arm_model=False`を宣言するため、ロックされたベンチマークプールでドラフトできます）。

| ドラフター | モデル | ピン | ライセンス | デフォルト |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | はい |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | — |

EmbeddingGemma 2は、そのモデルカードに記載されている方法で使用されます。つまり、各コマンドにタスクのプレフィックスを付与し、テキストのみを読み込み、`--dim 768|512|256`を使用します。ダウンロードを禁止するには、`HF_HUB_OFFLINE=1`を設定します。`plain-sight`は常にオフラインであり、`qwen3-vl`は`HF_HUB_OFFLINE`（`local_files_only=True`）を尊重し、リビジョンの不一致を拒否（`MODEL_REVISION_MISMATCH`）、VRAMガバナーを尊重し、プロジェクトごとのプロンプト（`drafter_prompt`または`--drafter-prompt`）を受け入れ、プロンプトと画像ごとの完全な生の出力を保存します。

**動作環境:** Python 3.11および3.12、WindowsおよびLinux。デフォルトではCPUを使用し、すべてのスレッドプールは制限されます（デフォルトは4スレッドですが、`--threads`で指定された場合は異なる）。モデルの実行時には、プロセスは通常よりも低い優先度になり、これにより、複数のマシンで共有できます。

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

### プロジェクトごとのキャプションルール（`caption-rules.json`）とプロビナンス

`caption-rules.json`を`<project>/caption-rules.json`に配置します（または`--caption-config path/to/rules.json`を渡します）。

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

- **ゲートチェック（モデル不使用、`description-review.jsonl`に報告）：** `prefix_missing`（キャプションが`prefix`で始まっていない）、`required_term_missing`（そのレコードのクラス — `class_field`または`class_pattern`から解決 — の`required_terms`のエントリ、または`required`の背景が欠けている）、`banned_word`（グローバル、背景、またはクラスの`banned_words`のフレーズが含まれている）、さらにプールの`exact_duplicate`と`near_duplicate`（Jaccard ≥ 0.85）。
- **コンポジション（`compose`または`report --compose`）：** `<out>/captions-manifest.json`（`--manifest-name`でファイル名を上書き可能）を、`style-dataset-lab`のlane-3 `run_dataset`形式`{"items":[{"image","sha256","caption",...}]}`で書き出します。`--project`の中には決して書き込みません。各キャプションは、`prefix`、そのままの`class`必須用語、任意の`backdrop`、および自由な可視コンテンツ部分（インポートされている場合は`seat`の編集、それ以外は承認済みの人間/サイドカーのテキスト — 未編集のドラフトは決して使いません）から組み立てられます。デフォルトでは、`compose`と`report --compose`は、モデル不使用のゲートでフラグが立てられたレコード**のみ**をレビューキューにドラフトします。`--all`を指定すると、フラグのない元のレコードの横に、任意の`draft_offer` / `draft_offer_writer`の提案も書き出しますが、人間やサイドカーのテキストは置き換えません。
- **出所記録：** `captions-manifest.json`と`description-review.jsonl`の両方にある最終キャプションはすべて、`config:<file>:<line>`、`drafter:<id>@<rev>:<prompt sha>`、`seat:<name>`、`record:<file>`、`sidecar:<file>`を用いたパートごとの`writers`リストを持ちます。ドラフトや提案を見たあとにシートが編集した場合は層が`captioner_assisted`、人間のレコードは`"human"`、`.txt`サイドカーは`"sidecar"`になります。
- **サイドカーのインポート（`sidecar-import`または`--sidecars`）：** 各画像を同じ語幹の`.txt`サイドカーとペアにし、`author_kind: "sidecar"`を割り当て（`"human"`には決してしません）、埋め込まれたComfyUIのPNG生成プロンプトがあれば抽出し、完全な説明ゲート（プロンプトのエコー、完全一致/Jaccard重複、`caption-rules.json`）を実行し、`.txt`サイドカーのない画像はテキストを作り出さずに報告し、インポートしたすべてのレコードとレビューファイルを厳密に`--out`の下にのみ書き出します。
- **タグによるプールロック、フェイルクローズ方式のリネームコピーガード、`pool_detection`ステータス：** `check_multimodal_pool`は、`realpath`のフォルダ名、または設定（`caption-rules.json`または`project.json`）内の`"multimodal_pool"`のいずれかがロックされたプールを指す場合に、`--project`とテキストコマンド（`text-index`、`text-dupes`、`text-search`、`lineage`）の両方でベンチマークロックを強制します。それらの名前に当てはまらないタグなしフォルダについては、スキャンが読み取る画像の正確なセット（シンボリックリンクとWindowsのディレクトリジャンクションを含む）を、空でないローカルマニフェスト（`DATASET_DESK_POOL_MANIFEST`、またはリポジトリの外に置かれた`~/.config/dataset-desk/config.json`の`pool_manifest`）または`--locked-pool-dir <path>`に対してハッシュ照合し、テキストコマンドでは`.txt`キャプションサイドカーの語幹もロックされたプールの画像の語幹と照合して、一致したプールと件数を示しつつフェイルクローズで拒否します（`PERM_LOCKED_POOL`）。すべてのコマンドは1つの`pool_detection`ステータスを解決し（設定された各プールにつき画像ハッシュが >= 1 ある`status: "on"`とソースごとの件数 — ハッシュ値は決して含めません — または警告テキスト付きの`status: "off"`で、警告は`stderr`に一度だけ出力されます）、`pool_detection`を`<out>/pool-detection.json`、`--out`配下のJSON/JSONL出力、および各Markdownレポートの1行の注記に記録します。
  - リネームコピーの検出は`sha256`で画像を照合するため、新しい画像のみで作られたコピー（ロックされたプールと共有する画像がないもの）も、リサイズ、再エンコード、またはメタデータを削除したコピーも`sha256`では一致しません。そのようなプロジェクトには、`caption-rules.json`または`project.json`で`"multimodal_pool"`のタグを付けてください。

`dataset-desk --help`はすべてのフラグをリストします。終了コード：`0`（正常）、`1`（ユーザーエラーまたは拒否）、`2`（予期しないエラー）。エラーは、`error <CODE>: <message>`と`hint:`の行とともに表示されます。`--debug`はスタックトレースを追加します。

**プロジェクトが読み取るレイアウト**は、[style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab)のものです。

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

`records/`がないプロジェクトでも、画像が`outputs/`の下のフォルダーにある限り、正常に動作します（`outputs/`の上部にあるファイルは連絡先シートとしてスキップされます）。すべての画像は「未記録」であり、重複チェックとルックチェックは引き続き実行されます。

## 脅威モデルと信頼

- **読み取るもの:** `--project`で指定するプロジェクトフォルダー：画像、`records/*.json`、ウェーブメタデータ、およびPNGファイルに埋め込まれた生成データ。テキストコマンドは、指定する`--source`フォルダーまたは`--rnd-data`チェックアウトを読み取ります。rnd-dataのプライベートコンパニオンとblobストアは拒否し、相対パスのみを出力します。
- **書き込むもの:** `--out`の下にのみ書き込みます：スキャンキャッシュ、埋め込みキャッシュ、レポート、`description-review.jsonl`、およびワークシート。プロジェクト内に`--out`がある場合は拒否します（`PERM_OUT_INSIDE_PROJECT`）、したがって、プロジェクトは変更されません。
- **決して触れないもの:** これらの2つのフォルダーの外にあるもの。どこにも何も削除しません。
- **ネットワーク:** ライトティアは決して使用しません。モデルバックエンドは、初回使用時にHugging Faceからピン留めされた重みをフェッチします。ただし、`HF_HUB_OFFLINE=1`が設定されている場合は除きます。ドラフターは、毎回`HF_HUB_OFFLINE=1`と`local_files_only=True`でロードします。リモートコードを実行するモデルはありません。
- **権限:** 通常のユーザーアカウント。GPUには、`--device`と`--grant` IDが必要なため、共有カードが誤って使用されることはありません。モデルをロードする前に、VRAMガバナーは、現在使用されているメモリとモデルの測定された必要量との合計が、`--vram-budget`（デフォルトは26,624 MiB）を下回ることを確認します。何も削除することはありません。`--wait`は、ロードが完了するまでポーリングを続けます。
- **秘密とテレメトリ:** 認証情報を読み取りません。また、データを収集または送信しません。
- **保護されたステップ:** 実行中のベンチマーク用にロックされたプロジェクトの場合、説明の埋め込みと、説明のモデルベースの受け入れは、`--allow-multimodal-pool`が渡されない限り拒否されます。これにより、ベンチマークのクエリが、測定するモデルによって影響を受けないようにします。

[SECURITY.md](SECURITY.md)に記載されているように、脆弱性を報告してください。

## ステータス

**v0.3.0は、公開プロトタイプです。** コードは、モデルライブラリの小さな代替物を使用して、オフラインでテストされます。
- **キャリブレーション:** スタジオは、RTX 5090上で独自のデータを使用して実行され、重複の閾値を設定します。これは仮のものであり、種類ごとに1つのプールに依存します。VRAMの数値は、CUDAコンテキストを含めた、保守的な全体カードの数値であり、RTX 5090で測定されています。他のカードやデータセットでは異なる場合があります。コードの閾値は、17個の配置されたコピーに基づいています。この方法とデータは、[ニュースレター第16号](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/)に記載されています。
- **大規模プロジェクト:** 重複、外れ値、およびルックチャネルは、制限されたブロックで処理されます。合成データセットでは、ほぼ重複する画像の検索とルックチャネルの処理に、10万枚の画像で数分しかかかりません。以前は数時間かかったり、メモリが不足したりしていました。記述ゲートのほぼ重複チェックは、正確なプレフィックスフィルターを使用するため、0.85に到達する可能性のある記述のみを比較します。最初のスキャンは、1枚の画像あたり約2倍高速で、結果は同じであり、`--threads`の下で並行して実行されます。
- **1.0の前にコマンドラインが変更される可能性があります。**

設計記録、その研究の根拠、およびレビュー履歴は、スタジオの研究ライブラリにあります: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)。

## 開発

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CIは、90％以上の行カバレッジを維持します。

## ライセンス

[MIT](LICENSE)。このパッケージに含まれていない、ロードできるモデルはそれぞれ独自のライセンスを持ちます。使用する前に、モデルカードを読んでください。

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
