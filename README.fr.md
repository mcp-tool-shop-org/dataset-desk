<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## Pourquoi

Un style LoRA apprend à partir de ses images d’entraînement et de leurs légendes. Deux défauts mineurs gâchent le tout :
- **Répétitions.** La même image enregistrée deux fois, ou deux rendus décalés de quelques pixels, enseignent la même image deux fois.
- **Descriptions qui ne sont pas des descriptions.** Une légende copiée à partir de l’invite de génération, ou une légende générique collée sur 54 images, n’apporte aucune information sur l’image. Pire encore, si ce texte est utilisé ultérieurement pour tester la recherche, il mesure le rappel de l’invite, et non la compréhension de l’image.

dataset-desk détecte les deux, dans un dossier de projet, sans le modifier. Il génère un rapport et une file d’attente de révision pour qu’une personne puisse agir. **Rien n’est supprimé automatiquement.**

## Ce qu’il fait

| Commande | Ce que vous obtenez |
|---|---|
| `scan` | Analyse le projet dans `scan.jsonl` et `scan-report.json` sous `--out` sans charger d’encodeur |
| `index` | Analyse le projet (enregistrements, métadonnées audio, données de génération PNG) et encode chaque image unique dans une mémoire cache sous `--out` |
| `dupes` | Copies identiques (sha256), quasi-doublons (un dHash de 256 bits tenant compte des miroirs) et doublons sémantiques (cosinus d’encodage), au format JSON. Sur un index DINOv2, il utilise une règle à deux signaux calibrée. Deux vues du même objet ne sont jamais associées. |
| `search` | Les images les plus proches d’une description (`--text`), d’une image (`--image`) ou des deux |
| `coverage` | Les meilleures images pour chaque sujet dans une liste de sujets souhaités, afin que les lacunes soient visibles |
| `report` | Un rapport Markdown : doublons, les images approuvées les moins typiques en termes de contenu et d’apparence, l’audit des descriptions et si chaque description trouve sa propre image |
| `sidecar-import` | Importe des dossiers d’images associées à des fichiers de légende `.txt` portant le même nom de base (`--sidecars` sur `index`/`report`/`compose` fait la même chose), exécute le filtre de description, signale les images sans fichier `.txt` sans inventer de texte et enregistre tout sous `--out` |
| `compose` | Compose les légendes d’entraînement finales dans `--out` à partir des règles par projet et des modifications importées (par défaut, seuls les enregistrements marqués sont ajoutés à la file d’attente de révision, ou propose des brouillons côte à côte avec `--all`, sans remplacer le texte humain ou le fichier de légende), en écrivant un manifeste de jeu de données de niveau 3 (`{"items":[{"image","sha256","caption"}]}`) avec la provenance par partie |
| `import-edits` | Réimporte les lignes modifiées de `description-worksheet.md` dans `--out/seat-edits.jsonl` avec un ID d’auteur `seat:<name>` |
| `text-index` | Encode les fichiers texte et code d’un dossier `--source` (ou d’une copie de données), en comptant les blocs avec le tokenizer du modèle |
| `text-dupes` | Fichiers identiques et quasi-identiques. `--task code` compare uniquement les scripts. |
| `text-search` | Les fichiers les plus proches d’une requête (`--text`). `--task code` utilise l’invite de récupération de code du modèle. |
| `lineage` | Versions du même script dans différentes exécutions, classées par similarité |

**Les seuils** proviennent de la calibration du studio sur ses propres données. Chacun d’eux est marqué comme provisoire dans le code, avec la mesure à partir de laquelle il a été obtenu :
- **Images :** sur un index DINOv2, une paire est un doublon lorsque `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Les autres index utilisent le dHash dans une plage de 26 bits, plus un cosinus ≥ 0,95.
- **Texte :** 0,98 avec bge-m3, et 0,95 pour le code avec EmbeddingGemma 2. Tout autre modèle nécessite `--semantic`.

Deux rendus du même objet provenant d’angles de caméra différents ne sont jamais des doublons. La vue provient du `provenance.asset_id` et du `render_id` de chaque enregistrement, ou des noms de fichiers tels que `<asset>__y<yaw>_e<elev>`.

**Le filtre de description** s’exécute sans aucun modèle. Une description est signalée lorsqu’elle :
- fait écho à son invite : elle est égale à l’invite ou se trouve à l’intérieur, partage une séquence de 6 mots ou plus avec celle-ci, ou 90 % ou plus de ses mots (80 % ou plus de ses mots de contenu) y apparaissent ;
- englobe la ligne de sujet de l’enregistrement : une séquence de 4 mots ou plus `canon.subject` apparaît comme une séquence continue de ses mots ;
- est trop courte : moins de 8 mots ou moins de 5 mots de contenu ;
- duplique une autre description dans l’ensemble : identique une fois la ponctuation supprimée, ou un Jaccard de mots de contenu de 0,85 ou plus ;
- est trop longue pour SigLIP2 : plus de 64 jetons (la seule vérification qui nécessite un tokenizer ; les enregistrements non vérifiés ne sont jamais marqués comme valides) ;
- viole les règles `caption-rules.json` par projet (lorsqu’elles sont présentes) : `prefix_missing`, `required_term_missing` ou `banned_word` (en plus des vérifications `exact_duplicate` et `near_duplicate` du pool).

Les enregistrements signalés sont envoyés à `description-review.jsonl` et à une **feuille de travail aveugle** : chaque image signalée à côté de 3 à 5 images sœurs visuellement les plus proches (choisies à partir de la similarité DINOv2 image uniquement lorsque mises en cache sous `--out`, sinon dHash ; jamais d’encodages de texte dans un pool verrouillé), sans invite, sans ligne de sujet et sans ancien texte visible, afin que la personne qui la réécrive décrive l’image. Un rédacteur facultatif (`--draft plain-sight` par défaut, ou `--draft qwen3-vl`) peut rédiger un premier brouillon à partir des pixels seuls dans la file d’attente de révision pour le siège (jamais directement dans la légende d’entraînement) ; la sortie brute complète du rédacteur est enregistrée par image dans `drafter-raw.jsonl`. Le fait de valider `--all` permet également d’ajouter des enregistrements non signalés sous forme de propositions côte à côte (`draft_offer` / `draft_offer_writer`) sans remplacer le texte humain ou le fichier de légende. Les brouillons modifiés par le siège sont enregistrés sous `captioner_assisted`.

**Le canal d’apparence** est composé de 30 nombres par image (palette, tonalité et texture à la résolution native), de sorte que « le rendu de celle-ci est incorrect » apparaît séparément de « le sujet de celle-ci est inhabituel ».

## Installation

```bash
pip install dataset-desk
```

Cela donne le **niveau de base** : uniquement numpy et Pillow, pas de torch. Les vérifications de doublons, le canal d’apparence, le filtre de description et le rapport fonctionnent tous dans ce niveau (`--backend none`).

Pour l’index d’encodage (recherche, couverture, doublons sémantiques, vérification de l’adéquation de la description), ajoutez les bibliothèques du modèle :

```bash
pip install "dataset-desk[models]"
```

Sous Windows avec une carte graphique NVIDIA, installez d’abord une version CUDA de torch à partir de `https://download.pytorch.org` PyPI est une version de torch pour Windows uniquement CPU.

Les modèles sont récupérés sur Hugging Face aux révisions fixées la première fois qu’ils sont utilisés, puis chargés à partir de votre cache. Chaque tâche a une valeur par défaut, et `--backend` en choisit une autre :

| Backend | Modèle | Valeur par défaut pour |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | texte |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | code (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

Rédacteurs d’images aveugles (`--draft`, `dataset_desk.registry.DRAFTER_REGISTRY` ; les deux déclarent `is_arm_model=False`, de sorte que les pools de référence verrouillés leur permettent de rédiger) :

| Rédacteur | Modèle | Fixer | Licence | Valeur par défaut |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | oui |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | — |

EmbeddingGemma 2 est utilisé comme indiqué dans sa fiche descriptive : un préfixe de tâche par commande, un chargement en texte seul pour le texte, et `--dim 768|512|256`. Définissez `HF_HUB_OFFLINE=1` pour interdire les téléchargements. `plain-sight` est toujours hors ligne ; `qwen3-vl` respecte `HF_HUB_OFFLINE` (`local_files_only=True`), refuse toute divergence de révision (`MODEL_REVISION_MISMATCH`), respecte le régulateur de VRAM, accepte un prompt par projet (`drafter_prompt` ou `--drafter-prompt`), et stocke le prompt et la sortie brute complète par image.

**Fonctionne sur :** Python 3.11 et 3.12, sur Windows et Linux. CPU par défaut, avec chaque pool de threads limité (4 threads, sauf si `--threads` en indique autrement), et l’exécution d’un modèle réduit la priorité du processus à une priorité inférieure à la normale, afin qu’il puisse partager une machine.

## Utilisation

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

### Règles de légende par projet (`caption-rules.json`) et provenance

Placez `caption-rules.json` à `<project>/caption-rules.json` (ou passez `--caption-config path/to/rules.json`) :

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

- **Contrôles de la porte (sans modèle, consignés dans `description-review.jsonl`) :** `prefix_missing` (la légende ne commence pas par `prefix`), `required_term_missing` (une entrée de `required_terms` pour la classe de l’enregistrement — résolue à partir de `class_field` ou `class_pattern` — ou un arrière-plan `required` est absente), `banned_word` (une expression de `banned_words` globale, d’arrière-plan ou de classe apparaît), ainsi que `exact_duplicate` et `near_duplicate` au niveau du pool (Jaccard ≥ 0.85).
- **Composition (`compose` ou `report --compose`) :** écrit `<out>/captions-manifest.json` (`--manifest-name` remplace le nom du fichier) au format lane-3 `run_dataset` de `style-dataset-lab` `{"items":[{"image","sha256","caption",...}]}` — jamais à l’intérieur de `--project`. Chaque légende est assemblée à partir de `prefix`, des termes requis de `class` repris tels quels, d’un `backdrop` facultatif et de la partie libre décrivant le contenu visible (la modification `seat` lorsqu’elle est importée, sinon le texte humain/sidecar approuvé — jamais un brouillon non modifié). Par défaut, `compose` et `report --compose` rédigent des brouillons **uniquement** pour les enregistrements signalés par la porte sans modèle, dans la file d’attente de révision ; `--all` écrit aussi des propositions facultatives `draft_offer` / `draft_offer_writer` à côté des originaux non signalés, sans remplacer le texte humain ou sidecar.
- **Enregistrement de provenance :** chaque légende finale, dans `captions-manifest.json` comme dans `description-review.jsonl`, porte une liste `writers` par partie utilisant `config:<file>:<line>`, `drafter:<id>@<rev>:<prompt sha>`, `seat:<name>`, `record:<file>` et `sidecar:<file>`, avec la strate `captioner_assisted` lorsqu’elle a été modifiée par un siège après avoir vu un brouillon ou une proposition, `"human"` pour les enregistrements humains et `"sidecar"` pour les fichiers sidecar `.txt`.
- **Import de sidecar (`sidecar-import` ou `--sidecars`) :** associe chaque image à son fichier sidecar `.txt` de même nom de base, attribue `author_kind: "sidecar"` (jamais `"human"`), extrait les prompts de génération ComfyUI intégrés aux PNG lorsqu’ils sont présents, exécute la porte de description complète (écho du prompt, doublons exacts/Jaccard et `caption-rules.json`), signale toute image sans sidecar `.txt` au lieu d’inventer du texte, et écrit tous les enregistrements importés et les fichiers de révision strictement sous `--out`.
- **Verrou du pool par étiquette, garde fail-closed contre les copies renommées et statut `pool_detection` :** `check_multimodal_pool` applique le verrou de référence à la fois sur `--project` et sur les commandes texte (`text-index`, `text-dupes`, `text-search`, `lineage`) lorsque le nom de dossier `realpath` ou `"multimodal_pool"` dans la configuration (`caption-rules.json` ou `project.json`) désigne un pool verrouillé. Pour les dossiers sans étiquette en dehors de ces noms, il calcule l’empreinte de l’ensemble exact des images lues par l’analyse (liens symboliques et jonctions de répertoire Windows inclus) et la compare à un manifeste local non vide (`DATASET_DESK_POOL_MANIFEST` ou `pool_manifest` dans `~/.config/dataset-desk/config.json`, conservé hors du dépôt) ou à `--locked-pool-dir <path>` — et, pour les commandes texte, vérifie aussi les noms de base des sidecars de légende `.txt` par rapport à ceux des images du pool verrouillé — en refusant en mode fail-closed (`PERM_LOCKED_POOL`) avec le pool correspondant et le nombre. Chaque commande résout un seul statut `pool_detection` (`status: "on"` avec >= 1 empreinte d’image par pool configuré et des décomptes par source — jamais les valeurs d’empreinte — ou `status: "off"` avec le texte d’avertissement, émis une seule fois sur `stderr`) et enregistre `pool_detection` dans `<out>/pool-detection.json`, dans les sorties JSON/JSONL sous `--out` et dans une note d’une ligne de chaque rapport Markdown.
  - La détection des copies renommées compare les images par `sha256` ; ainsi, ni une copie composée uniquement de nouvelles images (aucune partagée avec un pool verrouillé) ni une copie redimensionnée, réencodée ou dépourvue de métadonnées ne correspond par `sha256` ; étiquetez ces projets avec `"multimodal_pool"` dans `caption-rules.json` ou `project.json`.

`dataset-desk --help` liste tous les indicateurs. Codes de sortie : `0` ok, `1` une erreur utilisateur ou un refus, `2` une erreur inattendue. Les erreurs s’affichent sous la forme `error <CODE>: <message>` avec une ligne `hint:` ; `--debug` ajoute la trace de la pile.

**La disposition du projet qu’il lit** est celle de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) :

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un projet sans `records/` fonctionne également, tant que les images se trouvent dans des dossiers sous `outputs/` (les fichiers situés en haut de `outputs/` sont ignorés comme feuilles de contact) : chaque image est « non enregistrée », et les vérifications de doublons et de similarité sont toujours effectuées.

## Modèle de menace et confiance

- **Ce qu’il lit :** le dossier de projet que vous nommez avec `--project` : images, `records/*.json`, métadonnées d’ondes et les données de génération intégrées dans les fichiers PNG. Les commandes de texte lisent le dossier `--source` ou la branche `--rnd-data` que vous nommez. Elles refusent le fichier compagnon privé et les magasins de blobs de rnd-data, et elles n’affichent que les chemins relatifs.
- **Ce qu’il écrit :** uniquement sous `--out` : son cache d’analyse, les caches d’intégration, le rapport, `description-review.jsonl` et la feuille de calcul. Il refuse un `--out` à l’intérieur du projet (`PERM_OUT_INSIDE_PROJECT`), de sorte qu’un projet n’est jamais modifié.
- **Ce qu’il ne touche jamais :** rien en dehors de ces deux dossiers. Il ne supprime rien, nulle part.
- **Réseau :** la version légère ne l’utilise jamais. Un backend de modèle récupère ses poids fixes à partir de Hugging Face lors de la première utilisation, à moins que `HF_HUB_OFFLINE=1` ne soit défini ; le rédacteur se charge avec `HF_HUB_OFFLINE=1` et `local_files_only=True` à chaque fois. Aucun modèle n’exécute de code à distance.
- **Autorisations :** un compte d’utilisateur ordinaire. Une GPU nécessite `--device` plus un ID `--grant`, afin qu’une carte partagée ne soit pas utilisée par erreur. Avant de charger un modèle, un régulateur de VRAM vérifie que la mémoire déjà utilisée plus les besoins mesurés du modèle s’inscrivent dans `--vram-budget` (par défaut 26 624 MiB). Il ne supprime jamais rien ; `--wait` interroge jusqu’à ce que le chargement s’effectue.
- **Secrets et télémétrie :** il ne lit aucun identifiant et ne collecte ni n’envoie aucune télémétrie.
- **Étapes protégées :** pour les projets verrouillés pour un benchmark en cours, les intégrations de descriptions et toute acceptation des descriptions basée sur un modèle sont refusées, à moins que `--allow-multimodal-pool` ne soit transmis, afin que les requêtes du benchmark ne soient pas influencées par les modèles qu’il mesure.

Signalez une vulnérabilité comme décrit dans [SECURITY.md](SECURITY.md).

## Statut

**v0.3.0, un prototype public.** Le code est testé hors ligne avec de petits exemples pour les bibliothèques de modèles.
- **Calibration :** les exécutions du studio se font sur ses propres données, sur une RTX 5090, et définissent les seuils de duplication, qui sont provisoires et reposent sur un seul ensemble par type. Les chiffres de la VRAM sont des chiffres conservateurs pour la carte entière, contexte CUDA inclus, mesurés sur une RTX 5090 ; d’autres cartes et d’autres corpus peuvent différer. Les seuils de code reposent sur 17 copies implantées. La méthode et les données sont disponibles dans [le numéro 16 de la newsletter](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Grands projets :** les doublons, les valeurs aberrantes et le canal de recherche fonctionnent dans des blocs limités. Sur les ensembles synthétiques, la recherche de doublons et le canal de recherche prennent quelques minutes pour 100 000 images, alors qu’ils prenaient des heures ou que la mémoire était saturée. La vérification des doublons du filtre de description utilise un filtre de préfixe exact, de sorte qu’il ne compare que les descriptions qui pourraient atteindre 0,85. Le premier balayage est environ 2 fois plus rapide par image, avec des résultats identiques, et s’exécute en parallèle sous `--threads`.
- **Il faut s’attendre à ce que la ligne de commande change avant la version 1.0.**

Le document de conception, sa base de recherche et son historique d’évaluation se trouvent dans la bibliothèque de recherche du studio : [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Développement

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

L’intégration continue maintient une couverture de code à 90 % ou plus.

## Licence

[MIT](LICENSE). Les modèles qu’il peut charger ne font pas partie de ce paquet, et chacun est soumis à sa propre licence ; lisez la fiche du modèle avant de l’utiliser.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
