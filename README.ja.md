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
- **繰り返し:** 同じ画像を2回保存したり、数ピクセル異なる2つの画像をレンダリングしたりすると、同じ画像を2回学習させてしまいます。
- **説明になっていない説明:** 生成プロンプトからコピーされたキャプション、または54枚の画像に貼り付けられたキャプションは、モデルに画像に関する情報を何も伝えません。さらに悪いことに、そのテキストが後で検索のテストに使用された場合、それは画像の理解ではなく、プロンプトの再現率を測定します。

dataset-deskは、プロジェクトフォルダ内でこれら2つを検出し、変更を加えることなく、レポートとレビューキューを作成します。**自動的に削除されることはありません。**

## 機能

| コマンド | 出力 |
|---|---|
| `index` | プロジェクトをスキャンし（レコード、波形メタデータ、PNG生成データ）、すべてのユニークな画像を`--out`の下のキャッシュに埋め込みます。 |
| `dupes` | バイト単位で同一のコピー（sha256）、ほぼ同一のコピー（256ビットdHashで10ビット以内）、および意味的に同一のコピー（埋め込みコサイン）をJSON形式で出力します。 |
| `search` | 説明（`--text`）、画像（`--image`）、またはその両方に対して、最も近い画像を検索します。 |
| `coverage` | 必要な被写体リスト内の各被写体に対して、最適な画像を選択し、ギャップを表示します。 |
| `report` | Markdown形式のレポートを作成します。内容は、重複、コンテンツと見た目に基づいて最も一般的でない承認された画像、説明の監査、および各説明が独自の画像に対応しているかどうかです。 |

**説明ゲート**は、モデルを使用せずに実行されます。説明が以下の条件を満たす場合、フラグが立てられます。
- プロンプトをエコーしている：プロンプトと等しいか、プロンプト内に含まれているか、6語以上の連続した単語を共有しているか、または90％以上の単語（コンテンツ単語の80％以上）がプロンプトに含まれている。
- レコードの件名行をラップしている：4語以上の`canon.subject`が連続した単語として出現する。
- 短すぎる：8語未満または5語未満のコンテンツ単語が含まれている。
- セット内の別の説明と重複している：句読点を削除すると同一になるか、コンテンツ単語のJaccard係数が0.85以上である。
- SigLIP2に対して長すぎる：64トークンを超える（トークナイザーが必要な唯一のチェックであり、チェックされていないレコードは合格とマークされることはありません）。

フラグが立てられたレコードは、`description-review.jsonl`と**ブラインドワークシート**に送信されます。各フラグが立てられた画像の隣に、視覚的に最も近い3〜5つの画像を表示します。プロンプト、件名行、および古いテキストは表示されません。これにより、画像を書き直す担当者は、画像の内容を説明できます。オプションで、ドラフター（[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)、Florence-2）を使用して、ピクセルからのみ最初のドラフトを作成できます。ドラフトは`captioner_assisted`として記録され、担当者が編集します。

**ルックチャネル**は、画像ごとに30個の数値（ネイティブ解像度でのパレット、トーン、テクスチャ）です。これにより、「この画像のレンダリングがおかしい」という問題が、「この画像の被写体が奇妙だ」という問題とは別に表示されます。

## インストール

```bash
pip install "dataset-desk @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.0"
```

これにより、**ライトティア**が提供されます。numpyとPillowのみを使用し、torchは使用しません。重複チェック、ルックチャネル、説明ゲート、およびレポートはすべて、この環境で動作します（`--backend none`）。

埋め込みインデックス（検索、カバレッジ、意味的な重複、説明の適合性チェック）を使用するには、モデルライブラリを追加します。

```bash
pip install "dataset-desk[models] @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.0"
```

モデルは、最初にそれらが使用されるときに、Hugging Faceから固定されたリビジョンで取得され、その後はキャッシュからロードされます。例：[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)（`--backend eg2`）または[SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384)（`--backend siglip2`）。ダウンロードを禁止するには、`HF_HUB_OFFLINE=1`を設定します。plain-sightドラフターは常にオフラインで動作します。ダウンロードは行われません。

**実行環境:** Python 3.11および3.12、WindowsおよびLinux。デフォルトではCPUを使用し、すべてのスレッドプールは制限されます（4スレッド。ただし、`--threads`で指定された場合は例外）。モデルの実行時には、プロセスの優先度が通常よりも低くなります。これにより、同じマシンを共有できます。

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
```

`dataset-desk --help`は、すべてのフラグをリストします。終了コード：`0`（正常）、`1`（ユーザーエラーまたは拒否）、`2`（予期しないエラー）。エラーは、`error <CODE>: <message>`と`hint:`の行として出力されます。`--debug`は、スタックトレースを追加します。

**読み込むプロジェクトレイアウト**は、[style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab)のものです。

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

`records/`がないプロジェクトでも、画像が`outputs/`の下のフォルダに配置されていれば（`outputs/`の最上位にあるファイルは連絡先シートとしてスキップされます）、正常に動作します。すべての画像は「未記録」と見なされ、重複チェックとルックチェックは引き続き実行されます。

## 脅威モデルと信頼

- **読み込むもの:** `--project`で指定されたプロジェクトフォルダ：画像、`records/*.json`、波形メタデータ、およびPNGファイルに埋め込まれた生成データ。
- **書き込むもの:** `--out`の下にのみ書き込みます：スキャンキャッシュ、埋め込みキャッシュ、レポート、`description-review.jsonl`、およびワークシート。プロジェクト内の`--out`は拒否します（`PERM_OUT_INSIDE_PROJECT`）。プロジェクトは変更されません。
- **決して触れないもの:** これらの2つのフォルダ以外のもの。どこにも何も削除しません。
- **ネットワーク:** ライトティアはネットワークを使用しません。モデルバックエンドは、最初にモデルが使用されるときに、Hugging Faceから固定された重みをフェッチします。ただし、`HF_HUB_OFFLINE=1`が設定されている場合は例外です。ドラフターは、`HF_HUB_OFFLINE=1`と`local_files_only=True`を使用して、毎回ロードされます。どのモデルもリモートコードを実行しません。
- **権限:** 通常のユーザーアカウント。GPUには、`--device`と`--grant`のIDが必要です。これにより、共有カードが誤って使用されるのを防ぎます。
- **機密情報とテレメトリ:** 認証情報は読み込みません。また、データを収集または送信しません。
- **保護されたステップ:** 実行中のベンチマーク用にロックされたプロジェクトの場合、説明の埋め込みと、モデルに基づく説明の受け入れは、`--allow-multimodal-pool`が渡されない限り拒否されます。これにより、ベンチマークのクエリが、測定対象のモデルによって影響を受けることがなくなります。

[SECURITY.md]に記載されているように、脆弱性を報告してください。

## ステータス

**v0.1.0、パブリックプロトタイプ。** コードはオフラインでテストされています（142個のテスト、96％の行カバレッジ。モデルライブラリの小さな代替品を使用）。ただし、まだ実際のデータで調整されていません。ほぼ同一のコピーと意味的な重複のしきい値は、開始点です。1.0になる前に、コマンドラインが変更される可能性があります。

デザイン記録、その研究の根拠、およびレビュー履歴は、スタジオの研究ライブラリに保存されています：[rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)。

## 開発

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CIにより、コードカバレッジが90％以上を維持されます。

## ライセンス

[MIT](LICENSE)。このパッケージに含まれていないモデルがあり、それぞれのモデルには個別のライセンスが適用されます。使用する前に、各モデルの情報を確認してください。

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
