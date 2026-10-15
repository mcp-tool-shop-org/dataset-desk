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
- **Descriptions qui ne sont pas des descriptions.** Une légende copiée à partir de l’invite de génération, ou une légende générique collée sur 54 images, n’apporte aucune information au modèle sur l’image. Pire encore, si ce texte est utilisé ultérieurement pour tester la recherche, il mesure le rappel de l’invite, et non la compréhension de l’image.

dataset-desk détecte les deux, dans un dossier de projet, sans le modifier. Il génère un rapport et une file d’attente de révision pour qu’une personne puisse agir. **Rien n’est supprimé automatiquement.**

## Ce qu’il fait

| Commande | Ce que vous obtenez |
|---|---|
| `index` | Analyse le projet (enregistrements, métadonnées audio, données de génération PNG) et intègre chaque image unique dans une mémoire cache sous `--out` |
| `dupes` | Copies identiques au niveau de l’octet (sha256), quasi-doublons (dHash sur 256 bits dans un intervalle de 10 bits) et doublons sémantiques (cosinus d’intégration), au format JSON |
| `search` | Les images les plus proches d’une description (`--text`), d’une image (`--image`) ou des deux |
| `coverage` | Les meilleures images pour chaque sujet dans une liste de sujets souhaités, afin que les lacunes soient visibles |
| `report` | Un rapport Markdown : doublons, les images approuvées les moins typiques en termes de contenu et d’apparence, l’audit des descriptions et si chaque description trouve sa propre image |

**La validation des descriptions** s’exécute sans aucun modèle. Une description est signalée lorsqu’elle :
- fait écho à son invite : elle est égale à l’invite ou se trouve à l’intérieur, partage une séquence de 6 mots ou plus avec celle-ci, ou 90 % ou plus de ses mots (80 % ou plus de ses mots de contenu) s’y trouvent ;
- englobe le titre du sujet de l’enregistrement : une séquence de 4 mots ou plus `canon.subject` apparaît comme une séquence continue de ses mots ;
- est trop courte : moins de 8 mots ou moins de 5 mots de contenu ;
- duplique une autre description dans l’ensemble : identique une fois la ponctuation supprimée, ou un Jaccard des mots de contenu de 0,85 ou plus ;
- est trop longue pour SigLIP2 : plus de 64 jetons (la seule vérification qui nécessite un tokenizer ; les enregistrements non vérifiés ne sont jamais marqués comme valides).

Les enregistrements signalés sont envoyés à `description-review.jsonl` et à une **feuille de travail anonyme** : chaque image signalée à côté de 3 à 5 images visuellement les plus proches, sans invite, sans titre de sujet et sans ancien texte visible, afin que la personne qui la réécrive décrive l’image. Un rédacteur facultatif ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) peut rédiger une première ébauche à partir des pixels seuls ; les ébauches sont enregistrées sous `captioner_assisted`, et une personne les modifie.

**Le canal d’apparence** est composé de 30 nombres par image (palette, tonalité et texture à la résolution native), de sorte que « le rendu de celle-ci est incorrect » s’affiche séparément de « le sujet de celle-ci est inhabituel ».

## Installation

```bash
pip install "dataset-desk @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

Cela donne la **version allégée** : uniquement numpy et Pillow, pas de torch. Les vérifications de doublons, le canal d’apparence, la validation des descriptions et le rapport fonctionnent tous dans cette version (`--backend none`).

Pour l’index d’intégration (recherche, couverture, doublons sémantiques, vérification de l’adéquation de la description), ajoutez les bibliothèques de modèle :

```bash
pip install "dataset-desk[models] @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

Les modèles sont récupérés sur Hugging Face aux révisions fixées la première fois qu’ils sont utilisés, puis chargés à partir de votre cache : [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`--backend eg2`) ou [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) (`--backend siglip2`). Définissez `HF_HUB_OFFLINE=1` pour interdire les téléchargements. Le rédacteur plain-sight est toujours hors ligne : il ne télécharge jamais.

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

- **Ce qu’il lit :** le dossier de projet que vous nommez avec `--project` : images, `records/*.json`, métadonnées audio et les données de génération intégrées dans les fichiers PNG.
- **Ce qu’il écrit :** uniquement sous `--out` : sa mémoire cache d’analyse, les mémoires cache d’intégration, le rapport, `description-review.jsonl` et la feuille de travail. Il refuse un `--out` à l’intérieur du projet (`PERM_OUT_INSIDE_PROJECT`), de sorte qu’un projet n’est jamais modifié.
- **Ce qu’il ne touche jamais :** quoi que ce soit en dehors de ces deux dossiers. Il ne supprime rien, nulle part.
- **Réseau :** la version allégée ne l’utilise jamais. Un backend de modèle récupère ses poids fixés sur Hugging Face lors de la première utilisation, à moins que `HF_HUB_OFFLINE=1` ne soit défini ; le rédacteur se charge avec `HF_HUB_OFFLINE=1` et `local_files_only=True` à chaque fois. Aucun modèle n’exécute de code à distance.
- **Autorisations :** un compte d’utilisateur ordinaire. Une GPU nécessite `--device` plus un ID `--grant`, afin qu’une carte partagée ne soit pas utilisée par erreur.
- **Secrets et télémétrie :** il ne lit aucun identifiant et ne collecte ni n’envoie aucune télémétrie.
- **Étapes protégées :** sur les projets verrouillés pour un test de référence en cours, les intégrations de descriptions et toute acceptation de descriptions basée sur un modèle sont refusées, à moins que `--allow-multimodal-pool` ne soit transmis, afin que les requêtes du test de référence ne soient pas influencées par les modèles qu’il mesure.

Signalez une vulnérabilité comme décrit dans [SECURITY.md](SECURITY.md).

## État

**v0.1.2, un prototype public.** Le code est testé hors ligne (152 tests, 96 % de couverture des lignes, avec de petits éléments de substitution pour les bibliothèques de modèle), mais il n’a pas encore été calibré sur des données réelles : les seuils de quasi-doublons et de doublons sémantiques sont des points de départ. Attendez-vous à ce que la ligne de commande change avant la version 1.0.

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

[MIT](LICENSE). Les modèles qu’il peut charger ne font pas partie de ce paquet, et chacun est soumis à sa propre licence ; veuillez consulter la fiche descriptive du modèle avant de l’utiliser.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
