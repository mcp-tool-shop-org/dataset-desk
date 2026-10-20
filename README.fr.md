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

Un style LoRA apprend à partir de ses images d’entraînement et de leurs légendes. Deux défauts mineurs nuisent aux deux :
- **Répétitions.** La même image enregistrée deux fois, ou deux rendus décalés de quelques pixels, enseignent la même image deux fois.
- **Descriptions qui ne sont pas des descriptions.** Une légende copiée à partir de l’invite de génération, ou une légende générique collée sur 54 images, n’apporte aucune information sur l’image. Pire encore, si ce texte est utilisé ultérieurement pour tester la recherche, il mesure le rappel de l’invite, et non la compréhension de l’image.

dataset-desk détecte les deux, dans un dossier de projet, sans le modifier. Il génère un rapport et une file d’attente de révision pour qu’une personne puisse agir. **Rien n’est supprimé automatiquement.**

## Ce qu’il fait

| Commande | Ce que vous obtenez |
|---|---|
| `index` | Analyse le projet (enregistrements, métadonnées d’ondes, données de génération PNG) et intègre chaque image unique dans une mémoire cache sous `--out` |
| `dupes` | Copies identiques au niveau de l’octet (sha256), quasi-doublons (dHash 256 bits prenant en compte les miroirs) et doublons sémantiques (cosinus d’intégration), au format JSON. Sur un index DINOv2, il utilise une règle à deux signaux calibrée. Deux vues du même objet ne sont jamais associées. |
| `search` | Les images les plus proches d’une description (`--text`), d’une image (`--image`) ou des deux |
| `coverage` | Les meilleures images pour chaque sujet dans une liste de sujets souhaités, afin que les lacunes soient visibles |
| `report` | Un rapport Markdown : doublons, les images approuvées les moins typiques en termes de contenu et d’apparence, l’audit des descriptions et si chaque description trouve sa propre image |
| `text-index` | Intègre les fichiers texte et de code d’un dossier `--source` (ou d’une copie de données rnd), en comptant les blocs avec le tokenizer du modèle |
| `text-dupes` | Fichiers exacts et quasi-doublons. `--task code` compare uniquement les scripts. |
| `text-search` | Les fichiers les plus proches d’une requête (`--text`). `--task code` utilise l’invite de récupération de code du modèle. |
| `lineage` | Versions du même script dans différentes exécutions, classées par similarité |

**Les seuils** proviennent de la calibration du studio sur ses propres données. Chacun d’eux est marqué comme provisoire dans le code, avec la mesure à partir de laquelle il a été obtenu :
- **Images :** sur un index DINOv2, une paire est un doublon lorsque `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Les autres index utilisent le dHash dans une plage de 26 bits, plus un cosinus ≥ 0,95.
- **Texte :** 0,98 avec bge-m3, et 0,95 pour le code avec EmbeddingGemma 2. Tout autre modèle nécessite `--semantic`.

Deux rendus du même objet provenant d’angles de caméra différents ne sont jamais des doublons. La vue provient de `provenance.asset_id` et `render_id` de chaque enregistrement, ou de noms de fichiers tels que `<asset>__y<yaw>_e<elev>`.

**La porte de description** s’exécute sans aucun modèle. Une description est signalée lorsqu’elle :
- fait écho à son invite : elle est égale à l’invite ou se trouve à l’intérieur, partage une séquence de 6 mots ou plus avec celle-ci, ou 90 % ou plus de ses mots (80 % ou plus de ses mots de contenu) y apparaissent ;
- englobe la ligne de sujet de l’enregistrement : une séquence de 4 mots ou plus `canon.subject` apparaît comme une séquence contiguë de ses mots ;
- est trop courte : moins de 8 mots ou moins de 5 mots de contenu ;
- duplique une autre description dans l’ensemble : identique une fois la ponctuation supprimée, ou un Jaccard de mots de contenu de 0,85 ou plus ;
- est trop longue pour SigLIP2 : plus de 64 jetons (la seule vérification qui nécessite un tokenizer ; les enregistrements non vérifiés ne sont jamais marqués comme valides).

Les enregistrements signalés sont envoyés à `description-review.jsonl` et à une **feuille de travail aveugle** : chaque image signalée à côté de 3 à 5 images sœurs visuellement les plus proches, sans invite, sans ligne de sujet et sans ancien texte visible, de sorte que la personne qui la réécrit décrive l’image. Un rédacteur facultatif ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) peut rédiger une première ébauche à partir des seuls pixels ; les ébauches sont enregistrées sous `captioner_assisted`, et une personne les modifie.

**Le canal d’apparence** est composé de 30 nombres par image (palette, tonalité et texture à la résolution native), de sorte que « le rendu de celle-ci est incorrect » s’affiche séparément de « le sujet de celle-ci est inhabituel ».

## Installation

```bash
pip install dataset-desk
```

Cela donne le **niveau de base** : uniquement numpy et Pillow, pas de torch. Les vérifications de doublons, le canal d’apparence, la porte de description et le rapport fonctionnent tous dans celui-ci (`--backend none`).

Pour l’index d’intégration (recherche, couverture, doublons sémantiques, vérification de l’adéquation de la description), ajoutez les bibliothèques de modèle :

```bash
pip install "dataset-desk[models]"
```

Les modèles sont récupérés sur Hugging Face aux révisions fixées la première fois qu’ils sont utilisés, puis chargés à partir de votre cache. Chaque tâche a une valeur par défaut, et `--backend` en choisit une autre :

| Backend | Modèle | Valeur par défaut pour |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | texte |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | code (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

EmbeddingGemma 2 est utilisé de la manière indiquée dans sa fiche de modèle : un préfixe de tâche par commande, chargement uniquement du texte pour le texte et `--dim 768|512|256`. Définissez `HF_HUB_OFFLINE=1` pour interdire les téléchargements. Le rédacteur plain-sight est toujours hors ligne : il ne télécharge jamais.

**Fonctionne sur :** Python 3.11 et 3.12, sur Windows et Linux. CPU par défaut, avec chaque pool de threads limité (4 threads, sauf si `--threads` en indique autrement), et une exécution de modèle réduit le processus à une priorité inférieure à la normale, afin qu’il puisse partager une machine.

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
```

`dataset-desk --help` répertorie tous les indicateurs. Codes de sortie : `0` ok, `1` une erreur utilisateur ou un refus, `2` une erreur inattendue. Les erreurs s’affichent sous la forme `error <CODE>: <message>` avec une ligne `hint:` ; `--debug` ajoute la trace de la pile.

**La disposition du projet qu’il lit** est celle de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) :

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un projet sans `records/` fonctionne également, tant que les images se trouvent dans des dossiers sous `outputs/` (les fichiers situés en haut de `outputs/` sont ignorés en tant que feuilles de contact) : chaque image est « non enregistrée », et les vérifications de doublons et d’apparence sont toujours effectuées.

## Modèle de menace et confiance

- **Ce qu’il lit :** le dossier de projet que vous nommez avec `--project` : images, `records/*.json`, métadonnées des ondes et les données de génération intégrées dans les fichiers PNG. Les commandes de texte lisent le dossier `--source` ou la branche `--rnd-data` que vous nommez. Elles refusent les données privées et les dépôts de données de rnd-data, et elles n’affichent que les chemins relatifs.
- **Ce qu’il écrit :** uniquement dans `--out` : son cache d’analyse, les caches d’intégration, le rapport, `description-review.jsonl` et la feuille de calcul. Il refuse un `--out` à l’intérieur du projet (`PERM_OUT_INSIDE_PROJECT`), de sorte qu’un projet n’est jamais modifié.
- **Ce qu’il ne touche jamais :** tout ce qui se trouve en dehors de ces deux dossiers. Il ne supprime rien, nulle part.
- **Réseau :** la version allégée ne l’utilise jamais. Un moteur de modèle récupère ses poids pré-enregistrés sur Hugging Face lors de la première utilisation, à moins que `HF_HUB_OFFLINE=1` ne soit défini ; le module de génération est chargé avec `HF_HUB_OFFLINE=1` et `local_files_only=True` à chaque fois. Aucun modèle n’exécute de code à distance.
- **Autorisations :** un compte d’utilisateur ordinaire. Une GPU a besoin de `--device` plus un `--grant`, afin qu’une carte partagée ne soit pas utilisée par erreur. Avant de charger un modèle, un régulateur de VRAM vérifie que la mémoire déjà utilisée plus les besoins mesurés du modèle entrent dans `--vram-budget` (par défaut : 26 624 MiB). Il ne supprime jamais rien ; `--wait` effectue des vérifications jusqu’à ce que le chargement soit effectué.
- **Secrets et télémétrie :** il ne lit aucun identifiant et ne collecte ni n’envoie aucune télémétrie.
- **Étapes sécurisées :** pour les projets verrouillés pour un test de performance en cours, les intégrations de descriptions et toute acceptation de descriptions basée sur un modèle sont refusées, à moins que `--allow-multimodal-pool` ne soit transmis, afin que les requêtes du test de performance ne soient pas influencées par les modèles qu’il mesure.

Signalez une vulnérabilité comme décrit dans [SECURITY.md](SECURITY.md).

## État

**v0.2.0, un prototype public.** Le code est testé hors ligne : 188 tests et 95 % de couverture des lignes, avec de petits éléments de substitution pour les bibliothèques de modèles.
- **Calibrage :** les exécutions du studio sur ses propres données, sur une RTX 5090, ont défini les seuils de duplication et les valeurs de VRAM. Ils sont provisoires et reposent sur un seul ensemble par type. Les seuils de code reposent sur 17 copies implantées. La méthode et les données sont disponibles dans [newsletter issue 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Grands projets :** les doublons, les valeurs aberrantes et le canal de recherche fonctionnent dans des blocs délimités. Sur des ensembles synthétiques, la recherche de doublons et le canal de recherche prennent quelques minutes pour 100 000 images, alors qu’ils prenaient des heures ou manquaient de mémoire. La porte de description est environ 6 fois plus rapide. Sa vérification de Jaccard ralentit encore considérablement sur les très grands ensembles qui partagent de nombreux mots, et cette question de conception reste ouverte.
- **Attendez-vous à ce que la ligne de commande change avant la version 1.0.**

Le document de conception, sa base de recherche et son historique de révision sont disponibles dans la bibliothèque de recherche du studio : [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Développement

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI maintient la couverture des lignes à 90 % ou plus.

## Licence

[MIT](LICENSE). Les modèles qu’il peut charger ne font pas partie de ce paquet, et chacun est soumis à sa propre licence ; lisez la fiche du modèle avant de l’utiliser.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
