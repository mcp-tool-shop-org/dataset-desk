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

Un style LoRA apprend à partir de ses images d’entraînement et de leurs légendes. Deux défauts mineurs nuisent à son efficacité :
- **Répétitions.** La même image enregistrée deux fois, ou deux rendus décalés de quelques pixels, entraînent le modèle sur la même image deux fois.
- **Descriptions qui ne sont pas des descriptions.** Une légende copiée à partir de l’invite de génération, ou une légende générique collée sur 54 images, n’apporte aucune information au modèle sur l’image. Pire encore, si ce texte est utilisé ultérieurement pour tester la recherche, il mesure la capacité de rappel de l’invite, et non la compréhension de l’image.

dataset-desk détecte les deux, dans un dossier de projet, sans le modifier. Il génère un rapport et une liste de révisions pour qu’une personne puisse agir. **Rien n’est supprimé automatiquement.**

## Ce qu’il fait

| Commande | Ce que vous obtenez |
|---|---|
| `index` | Analyse le projet (enregistrements, métadonnées audio, données de génération PNG) et intègre chaque image unique dans une mémoire cache sous `--out` |
| `dupes` | Copies identiques au niveau de l’octet (sha256), quasi-doublons (dHash sur 256 bits, avec une différence de 10 bits maximum) et doublons sémantiques (similarité cosinus de l’intégration), au format JSON |
| `search` | Les images les plus proches d’une description (`--text`), d’une image (`--image`) ou des deux |
| `coverage` | Les meilleures images pour chaque sujet d’une liste de sujets souhaités, afin de mettre en évidence les lacunes |
| `report` | Un rapport au format Markdown : doublons, les images approuvées les moins typiques en termes de contenu et d’apparence, l’audit des descriptions et si chaque description correspond à sa propre image |

**Le filtre de description** fonctionne sans aucun modèle. Une description est signalée lorsqu’elle :
- fait écho à son invite : elle est identique à l’invite ou est incluse dans celle-ci, partage une séquence de 6 mots ou plus avec celle-ci, ou 90 % ou plus de ses mots (80 % ou plus de ses mots de contenu) y apparaissent ;
- englobe la ligne de sujet de l’enregistrement : une séquence de 4 mots ou plus `canon.subject` apparaît comme une séquence continue de ses mots ;
- est trop courte : moins de 8 mots ou moins de 5 mots de contenu ;
- duplique une autre description dans l’ensemble : identique une fois la ponctuation supprimée, ou un indice de Jaccard des mots de contenu de 0,85 ou plus ;
- est trop longue pour SigLIP2 : plus de 64 jetons (la seule vérification qui nécessite un tokenizer ; les enregistrements non vérifiés ne sont jamais marqués comme valides).

Les enregistrements signalés sont envoyés à `description-review.jsonl` et à une **feuille de travail anonyme** : chaque image signalée est affichée à côté de 3 à 5 images visuellement les plus proches, sans invite, sans ligne de sujet et sans ancien texte visible, afin que la personne qui la réécrive puisse décrire l’image. Un rédacteur facultatif ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) peut rédiger une première version à partir des pixels seuls ; les brouillons sont enregistrés sous `captioner_assisted`, et une personne les modifie.

**Le canal d’apparence** est composé de 30 nombres par image (palette, tonalité et texture à la résolution native), de sorte que « le rendu de celle-ci est incorrect » s’affiche séparément de « le sujet de celle-ci est inhabituel ».

## Installation

```bash
pip install dataset-desk
```

Cela donne la **version allégée** : uniquement numpy et Pillow, pas de torch. Les vérifications de doublons, le canal d’apparence, le filtre de description et le rapport fonctionnent tous dans cette version (`--backend none`).

Pour l’index d’intégration (recherche, couverture, doublons sémantiques, vérification de l’adéquation de la description), ajoutez les bibliothèques de modèle :

```bash
pip install "dataset-desk[models]"
```

Les modèles sont récupérés sur Hugging Face aux versions fixes lors de leur première utilisation, puis sont chargés à partir de votre cache : [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`--backend eg2`) ou [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) (`--backend siglip2`). Définissez `HF_HUB_OFFLINE=1` pour interdire les téléchargements. Le rédacteur plain-sight est toujours hors ligne : il ne télécharge jamais.

**Fonctionne sur :** Python 3.11 et 3.12, sur Windows et Linux. CPU par défaut, avec chaque pool de threads limité (4 threads, sauf si `--threads` en indique autrement), et l’exécution d’un modèle réduit la priorité du processus à une priorité inférieure à la normale, afin qu’il puisse partager une machine.

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
```

`dataset-desk --help` répertorie tous les indicateurs. Codes de sortie : `0` ok, `1` une erreur utilisateur ou un refus, `2` une erreur inattendue. Les erreurs s’affichent sous la forme `error <CODE>: <message>` avec une ligne `hint:` ; `--debug` ajoute la trace de la pile.

**La disposition du projet qu’il lit** est celle de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab) :

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un projet sans `records/` fonctionne également, tant que les images se trouvent dans des dossiers sous `outputs/` (les fichiers situés en haut de `outputs/` sont ignorés car il s’agit de feuilles de contact) : chaque image est « non enregistrée », et les vérifications de doublons et d’apparence sont toujours effectuées.

## Modèle de menace et confiance

- **Ce qu’il lit :** le dossier de projet que vous nommez avec `--project` : images, `records/*.json`, métadonnées audio et les données de génération intégrées dans les fichiers PNG.
- **Ce qu’il écrit :** uniquement sous `--out` : sa mémoire cache d’analyse, les mémoires cache d’intégration, le rapport, `description-review.jsonl` et la feuille de travail. Il refuse un `--out` à l’intérieur du projet (`PERM_OUT_INSIDE_PROJECT`), de sorte qu’un projet n’est jamais modifié.
- **Ce qu’il ne touche jamais :** tout ce qui se trouve en dehors de ces deux dossiers. Il ne supprime rien, nulle part.
- **Réseau :** la version allégée ne l’utilise jamais. Un backend de modèle récupère ses poids fixes sur Hugging Face lors de sa première utilisation, sauf si `HF_HUB_OFFLINE=1` est défini ; le rédacteur est chargé avec `HF_HUB_OFFLINE=1` et `local_files_only=True` à chaque fois. Aucun modèle n’exécute de code à distance.
- **Autorisations :** un compte d’utilisateur ordinaire. Une carte graphique nécessite `--device` plus un ID `--grant`, afin qu’une carte partagée ne soit pas utilisée par erreur.
- **Secrets et télémétrie :** il ne lit aucun identifiant et ne collecte ni n’envoie aucune télémétrie.
- **Étapes protégées :** sur les projets verrouillés pour un test de référence en cours, les intégrations de description et toute acceptation de description basée sur un modèle sont refusées, sauf si `--allow-multimodal-pool` est transmis, afin que les requêtes du test de référence ne soient pas influencées par les modèles qu’il mesure.

Signalez une vulnérabilité comme décrit dans [SECURITY.md](SECURITY.md).

## État

**v0.1.3, un prototype public.** Le code est testé hors ligne (152 tests, 96 % de couverture des lignes, avec de petits éléments de substitution pour les bibliothèques de modèle), et une première calibration sur les données du studio a classé les modèles par tâche, mais n’a pas encore défini les seuils : les seuils de quasi-doublons et de doublons sémantiques ne sont encore que des points de départ. Attendez-vous à ce que la ligne de commande change avant la version 1.0.

Les documents de conception, les données de recherche et l’historique des révisions sont conservés dans la bibliothèque de recherche du studio : [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Développement

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

L’intégration continue garantit une couverture du code d’au moins 90 %.

## Licence

[MIT](LICENSE). Les modèles qu’il peut charger ne font pas partie de ce paquet, et chacun est soumis à sa propre licence ; lisez la fiche descriptive du modèle avant de l’utiliser.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
