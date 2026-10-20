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

## なぜ

スタイルLoRAは、トレーニング画像とそのキャプションから学習します。以下の2つの問題が両方を台無しにします。
- **繰り返し:** 同じ画像を2回保存したり、数ピクセル異なる2つの画像をレンダリングしたりすると、同じ画像を2回学習させてしまいます。
- **説明ではない説明:** 生成プロンプトからコピーされたキャプション、または54枚の画像に貼り付けられたキャプションは、モデルに画像に関する情報を何も伝えません。さらに悪いことに、そのテキストが後で検索のテストに使用された場合、それは画像の理解ではなく、プロンプトの再現率を測定します。

dataset-deskは、プロジェクトフォルダ上でこれら2つを検出し、変更することなく実行します。レポートとレビューキューを作成し、担当者が対応できるようにします。**自動的に削除されることはありません。**

## 機能

| コマンド | 結果 |
|---|---|
| `index` | プロジェクトをスキャンし（レコード、波形メタデータ、PNG生成データ）、すべてのユニークな画像を`--out`の下のキャッシュに埋め込みます。 |
| `dupes` | バイト単位で同一のコピー（sha256）、ほぼ同一のコピー（ミラーを考慮した256ビットdHash）、および意味的に同一のコピー（埋め込みコサイン）、JSON形式で出力します。DINOv2インデックスでは、調整された2つのシグナルルールを使用します。同じオブジェクトの2つのビューは決してペアリングされません。 |
| `search` | 説明（`--text`）、画像（`--image`）、またはその両方に対して、最も近い画像 |
| `coverage` | 必要な被写体リスト内の各被写体に対して、最適な画像を表示し、ギャップを明らかにします。 |
| `report` | Markdown形式のレポート：重複、コンテンツと外観に基づいて最も一般的でない承認された画像、説明の監査、および各説明が独自の画像を見つけられるかどうか。 |
| `text-index` | テキストファイルとコードファイル（`--source`フォルダまたはrnd-dataチェックアウト）を埋め込み、モデル自身のトークナイザーを使用してチャンクをカウントします。 |
| `text-dupes` | 正確に同一またはほぼ同一のファイル。`--task code`はスクリプトのみを比較します。 |
| `text-search` | クエリ（`--text`）に対して最も近いファイル。`--task code`はモデルのコード検索プロンプトを使用します。 |
| `lineage` | 実行間で、類似度に基づいてランク付けされた同じスクリプトのバージョン。 |

**閾値**は、スタジオが独自のデータで実行したキャリブレーションから得られます。それぞれがコード内で一時的なものとしてマークされ、測定値が記録されます。
- **画像:** DINOv2インデックスでは、ペアが重複するのは`(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`の場合です。他のインデックスでは、26ビット以内のdHashと、コサインが0.95以上の場合に使用します。
- **テキスト:** bge-m3では0.98、EmbeddingGemma 2ではコードに対して0.95。他のモデルでは、`--semantic`が必要です。

異なるカメラアングルから見た同じオブジェクトの2つのレンダリングは、決して重複しません。ビューは、各レコードの`provenance.asset_id`と`render_id`、またはファイル名（例：`<asset>__y<yaw>_e<elev>`）から取得されます。

**説明ゲート**は、モデルを使用せずに実行されます。説明が以下の条件を満たす場合、フラグが立てられます。
- プロンプトをエコーしている：プロンプトと等しいか、プロンプト内に含まれているか、6語以上の連続した単語を共有しているか、または90％以上の単語（コンテンツ単語の80％以上）がプロンプトに含まれている。
- レコードの件名行をラップしている：4語以上の`canon.subject`が、連続した単語として表示される。
- 短すぎる：8語未満または5語未満のコンテンツ単語。
- セット内の別の説明と重複している：句読点を削除すると同一であるか、コンテンツ単語のJaccard係数が0.85以上。
- SigLIP2に対して長すぎる：64トークンを超える（トークナイザーが必要な唯一のチェックであり、チェックされていないレコードは合格としてマークされることはありません）。

フラグが立てられたレコードは、`description-review.jsonl`と**ブラインドワークシート**に送信されます。各フラグが立てられた画像の隣に、視覚的に最も近い3〜5つの画像を表示し、プロンプト、件名行、および古いテキストを表示しないようにすることで、再作成する担当者が画像を説明できるようにします。オプションのドラフター（[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)、Florence-2）は、ピクセルのみから最初のドラフトを作成できます。ドラフトは`captioner_assisted`として記録され、担当者が編集します。

**外観チャンネル**は、画像ごとに30個の数値（ネイティブ解像度でのパレット、トーン、およびテクスチャ）であり、「この画像のレンダリングがおかしい」という問題が、「この画像の被写体が奇妙だ」という問題とは別に表示されます。

## インストール

```bash
pip install dataset-desk
```

これにより、**ライトティア**が提供されます。numpyとPillowのみを使用し、torchは使用しません。重複チェック、外観チャンネル、説明ゲート、およびレポートはすべて、これらで機能します（`--backend none`）。

埋め込みインデックス（検索、カバレッジ、意味的な重複、説明の適合性チェック）を使用するには、モデルライブラリを追加します。

```bash
pip install "dataset-desk[models]"
```

WindowsでNVIDIA GPUを使用する場合は、最初に`https://download.pytorch.org`からtorchのCUDAビルドをインストールします。PyPIのWindows用のtorchはCPU専用です。

モデルは、最初にそれらが使用されるときに、固定されたリビジョンでHugging Faceからフェッチされ、その後はキャッシュからロードされます。各タスクにはデフォルトがあり、`--backend`が別のものを選択します。

| バックエンド | モデル | デフォルト |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | テキスト |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | コード（`--task code`、`lineage`） |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

EmbeddingGemma 2は、そのモデルカードに記載されている方法で使用されます。コマンドごとにタスクプレフィックス、テキストのみのロード（テキスト用）、および`--dim 768|512|256`。`HF_HUB_OFFLINE=1`を設定して、ダウンロードを禁止します。plain-sightドラフターは常にオフラインです。ダウンロードは行いません。

**実行環境:** Python 3.11および3.12、WindowsおよびLinux。デフォルトではCPUを使用し、すべてのスレッドプールは制限されます（4スレッド、ただし`--threads`で指定された場合はそれ以外）。モデルの実行では、プロセスを通常よりも低い優先度まで下げ、マシンを共有できるようにします。

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
```

`dataset-desk --help`は、すべてのフラグをリストします。終了コード：`0`（正常）、`1`（ユーザーエラーまたは拒否）、`2`（予期しないエラー）。エラーは、`error <CODE>: <message>`と`hint:`の行とともに表示されます。`--debug`は、スタックトレースを追加します。

**読み込むプロジェクトレイアウト**は、[style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab)のものです。

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

`records/`がないプロジェクトでも、画像が`outputs/`の下のフォルダに配置されている限り、機能します（`outputs/`の最上位にあるファイルは、コンタクトシートとしてスキップされます）。すべての画像は「未記録」であり、重複チェックと外観チェックは引き続き実行されます。

## 脅威モデルと信頼

- **読み込むもの:** `--project`という名前を付けたプロジェクトフォルダー：画像、`records/*.json`、波形メタデータ、およびPNGファイルに埋め込まれた生成データ。テキストコマンドは、名前を付けた`--source`フォルダーまたは`--rnd-data`チェックアウトを読み込みます。rnd-dataのプライベートコンパニオンとblobストアは拒否し、相対パスのみを出力します。
- **書き込むもの:** `--out`の下でのみ：スキャンキャッシュ、埋め込みキャッシュ、レポート、`description-review.jsonl`、およびワークシート。プロジェクト内（`PERM_OUT_INSIDE_PROJECT`）に`--out`を書き込むことは許可されていません。そのため、プロジェクトは変更されません。
- **決して触らないもの:** これらの2つのフォルダーの外にあるもの。どこでも何も削除しません。
- **ネットワーク:** ライトティアは決して使用しません。モデルバックエンドは、初回使用時にHugging Faceから固定された重みをフェッチします。ただし、`HF_HUB_OFFLINE=1`が設定されている場合は除きます。ドラフターは、毎回`HF_HUB_OFFLINE=1`と`local_files_only=True`で読み込まれます。どのモデルもリモートコードを実行しません。
- **権限:** 通常のユーザーアカウント。GPUには`--device`と`--grant` IDが必要です。そのため、共有カードが誤って使用されることはありません。モデルを読み込む前に、VRAMガバナーは、すでに使用されているメモリとモデルの測定された必要量が`--vram-budget`（デフォルトは26,624 MiB）に収まるかどうかを確認します。何も強制的に削除することはありません。`--wait`がロードが完了するまでポーリングを続けます。
- **機密情報とテレメトリ:** 認証情報を読み取りません。また、テレメトリを収集または送信しません。
- **保護されたステップ:** 実行中のベンチマーク用にロックされたプロジェクトでは、説明の埋め込みと、モデルに基づく説明の受け入れは、`--allow-multimodal-pool`が渡されない限り拒否されます。そのため、ベンチマークのクエリは、測定対象のモデルによって影響を受けません。

[SECURITY.md](SECURITY.md)に記載されているように、脆弱性を報告してください。

## ステータス

**v0.2.1、公開プロトタイプ。** コードはオフラインでテストされています：197のテストと96％の行カバレッジ。モデルライブラリの小さな代替が使用されています。
- **キャリブレーション:** スタジオは、RTX 5090上で独自のデータで実行され、重複のしきい値を設定します。これは暫定的なものであり、1つのプール/種類に基づいています。VRAMの数値は、CUDAコンテキストを含め、保守的な全体カードの数値であり、RTX 5090で測定されています。他のカードやコーパスでは異なる場合があります。コードのしきい値は、17個の配置されたコピーに基づいています。メソッドとデータは、[ニュースレター第16号](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/)にあります。
- **大規模なプロジェクト:** 重複、外れ値、およびルックチャネルは、境界のあるブロックで機能します。合成データセットでは、ほぼ重複する検索とルックチャネルは、100,000枚の画像で数分で完了し、以前は数時間かかったり、メモリが不足したりしていました。説明ゲートのほぼ重複するチェックは、正確なプレフィックスフィルターを使用するため、0.85に到達する可能性のある説明のみを比較します。最初のスキャンは、1枚の画像あたり約2倍高速で、結果は同じであり、`--threads`の下で並行して実行されます。
- **1.0の前にコマンドラインが変更される可能性があります。**

設計記録、その研究の根拠、およびそのレビュー履歴は、スタジオの研究ライブラリにあります：[rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)。

## 開発

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CIは、行カバレッジを90％以上に維持します。

## ライセンス

[MIT](LICENSE)。読み込むことができるモデルは、このパッケージの一部ではなく、それぞれ独自のライセンスを持っています。使用する前に、モデルカードを読んでください。

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
