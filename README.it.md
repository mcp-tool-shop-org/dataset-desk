<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## Perché

Uno stile LoRA apprende dalle immagini di addestramento e dalle relative didascalie. Due difetti minori compromettono il risultato:
- **Ripetizioni.** La stessa immagine salvata due volte, o due rendering leggermente diversi, fanno sì che la stessa immagine venga "vista" due volte.
- **Descrizioni che non sono descrizioni.** Una didascalia copiata dal prompt di generazione, o una regola applicata a 54 immagini, non fornisce al modello alcuna informazione sull'immagine. Peggio ancora, se questo testo viene successivamente utilizzato per testare la ricerca, misura la capacità di recupero del prompt, non la comprensione dell'immagine.

dataset-desk individua entrambi i problemi all'interno di una cartella di progetto, senza modificarla. Genera un report e una coda di revisione per l'intervento di un operatore. **Niente viene rimosso automaticamente.**

## Cosa fa

| Comando | Risultato |
|---|---|
| `scan` | Scansiona il progetto in `scan.jsonl` e `scan-report.json` sotto `--out` senza caricare un codificatore. |
| `index` | Scansiona il progetto (record, metadati delle onde, dati di generazione PNG) e codifica ogni immagine univoca in una cache sotto `--out`. |
| `dupes` | Copie identiche (sha256), duplicati quasi identici (un dHash a 256 bit che tiene conto delle immagini speculari) e duplicati semantici (somiglianza coseno), in formato JSON. Su un indice DINOv2, utilizza una regola calibrata a due segnali. Due viste dello stesso oggetto non vengono mai accoppiate. |
| `search` | Le immagini più simili a una descrizione (`--text`), a un'immagine (`--image`) o a entrambe. |
| `coverage` | Le immagini migliori per ciascun soggetto in un elenco di soggetti desiderati, in modo che le lacune siano evidenti. |
| `report` | Un report in formato Markdown: duplicati, le immagini approvate meno tipiche in base al contenuto e all'aspetto, la verifica delle descrizioni e se ciascuna descrizione corrisponde alla propria immagine. |
| `sidecar-import` | Importa cartelle di immagini accoppiate con file di didascalia con lo stesso prefisso `.txt` (`--sidecars` su `index`/`report`/`compose` fa la stessa cosa), esegue il controllo delle descrizioni, segnala le immagini prive di un file di didascalia `.txt` senza inventare testo e salva tutto sotto `--out`. |
| `compose` | Compone le didascalie di addestramento finali in `--out` a partire da regole per progetto e modifiche importate (per impostazione predefinita, solo i record contrassegnati vengono inseriti nella coda di revisione, oppure offre bozze affiancate con `--all`, senza sostituire il testo umano o del file di didascalia), creando un file manifest del set di dati di livello 3 (`{"items":[{"image","sha256","caption"}]}`) con informazioni sulla provenienza per ciascuna parte. |
| `import-edits` | Reimporta le righe modificate da `description-worksheet.md` in `--out/seat-edits.jsonl` con un ID scrittore `seat:<name>`. |
| `text-index` | Codifica i file di testo e codice di una cartella `--source` (o di un checkout rnd-data), contando i blocchi con il tokenizer del modello. |
| `text-dupes` | File identici e quasi identici. `--task code` confronta solo gli script. |
| `text-search` | I file più simili a una query (`--text`). `--task code` utilizza il prompt di recupero del codice del modello. |
| `lineage` | Versioni dello stesso script tra le diverse esecuzioni, ordinate per somiglianza. |

**Le soglie** derivano dalla calibrazione del modello sui propri dati. Ognuna di esse è contrassegnata come provvisoria nel codice, con l'indicazione della misurazione da cui deriva:
- **Immagini:** su un indice DINOv2, una coppia è considerata un duplicato quando `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Altri indici utilizzano il valore dHash entro 26 bit, più il coseno ≥ 0,95.
- **Testo:** 0,98 con bge-m3 e 0,95 per il codice con EmbeddingGemma 2. Qualsiasi altro modello richiede `--semantic`.

Due rendering dello stesso oggetto da angolazioni diverse non sono mai considerati duplicati. La vista deriva dal `provenance.asset_id` e dal `render_id` di ciascun record, oppure dai nomi dei file come `<asset>__y<yaw>_e<elev>`.

**Il controllo delle descrizioni** viene eseguito senza l'utilizzo di alcun modello. Una descrizione viene contrassegnata quando:
- ripete il prompt: è uguale al prompt o è contenuto in esso, condivide una sequenza di 6 o più parole con esso, oppure il 90% o più delle sue parole (80% o più delle parole significative) compaiono in esso;
- include la riga dell'oggetto del record: una sequenza di 4 o più parole `canon.subject` compare come una sequenza continua di parole;
- è troppo breve: meno di 8 parole o meno di 5 parole significative;
- duplica un'altra descrizione nel set: è identica dopo aver rimosso la punteggiatura, oppure ha un indice di Jaccard delle parole significative pari a 0,85 o superiore;
- è troppo lunga per SigLIP2: più di 64 token (l'unico controllo che richiede un tokenizer; i record non controllati non vengono mai contrassegnati come validi);
- viola le regole per progetto `caption-rules.json` (se presenti): `prefix_missing`, `required_term_missing` o `banned_word` (insieme ai controlli del pool `exact_duplicate` e `near_duplicate`).

I record contrassegnati vengono inviati a `description-review.jsonl` e a una **scheda di lavoro anonima**: ogni immagine contrassegnata è affiancata da 3-5 immagini visivamente più simili (selezionate dall'indice di somiglianza delle sole immagini DINOv2 quando memorizzate nella cache sotto `--out`, altrimenti dHash; mai utilizzando le codifiche del testo in un pool bloccato), senza prompt, senza riga dell'oggetto e senza il vecchio testo visualizzato, in modo che la persona che la riscrive descriva l'immagine. Un redattore opzionale (`--draft plain-sight` per impostazione predefinita, oppure `--draft qwen3-vl`) può scrivere una prima bozza a partire dai soli pixel e inserirla nella coda di revisione per l'operatore (mai direttamente nella didascalia di addestramento); l'output completo del redattore viene salvato per ogni immagine in `drafter-raw.jsonl`. Se un record supera il controllo `--all`, anche i record non contrassegnati vengono inseriti come offerte affiancate (`draft_offer` / `draft_offer_writer`) senza sostituire il testo umano o del file di didascalia. Le bozze modificate dall'operatore vengono registrate come `captioner_assisted`.

**Il canale dell'aspetto** è costituito da 30 numeri per immagine (palette, tonalità e texture alla risoluzione nativa), in modo che "il rendering di questa immagine è errato" sia distinguibile da "il soggetto di questa immagine è strano".

## Installazione

```bash
pip install dataset-desk
```

Questo fornisce la **versione lite**: solo numpy e Pillow, senza torch. I controlli dei duplicati, il canale dell'aspetto, il controllo delle descrizioni e il report funzionano in questa versione (`--backend none`).

Per l'indice di codifica (ricerca, copertura, duplicati semantici, controllo della corrispondenza delle descrizioni), aggiungere le librerie del modello:

```bash
pip install "dataset-desk[models]"
```

Su Windows con una GPU NVIDIA, installare prima una versione di torch compilata per CUDA da `https://download.pytorch.org`; la versione di torch di PyPI per Windows è solo CPU.

I modelli vengono scaricati da Hugging Face alla prima esecuzione, utilizzando le versioni specificate, e quindi caricati dalla cache. Ogni attività ha un valore predefinito e `--backend` ne seleziona un altro:

| Backend | Modello | Predefinito per |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | testo |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | codice (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

Redattori di immagini anonimi (`--draft`, `dataset_desk.registry.DRAFTER_REGISTRY`; entrambi dichiarano `is_arm_model=False`, quindi i pool di benchmark bloccati consentono loro di redigere):

| Redattore | Modello | Fissa | Licenza | Predefinito |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | sì |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | — |

EmbeddingGemma 2 viene utilizzato come indicato nella relativa scheda del modello: un prefisso di attività per ogni comando, caricamento solo di testo per il testo e `--dim 768|512|256`. Imposta `HF_HUB_OFFLINE=1` per impedire i download. `plain-sight` è sempre offline; `qwen3-vl` rispetta `HF_HUB_OFFLINE` (`local_files_only=True`), rifiuta qualsiasi discrepanza di revisione (`MODEL_REVISION_MISMATCH`), rispetta il regolatore VRAM, accetta un prompt per progetto (`drafter_prompt` o `--drafter-prompt`) e memorizza il prompt e l'output completo per ogni immagine.

**Funziona su:** Python 3.11 e 3.12, su Windows e Linux. CPU per impostazione predefinita, con ogni pool di thread limitato (4 thread a meno che `--threads` non indichi diversamente), e l'esecuzione del modello riduce la priorità del processo a sotto il livello normale, in modo che possa condividere una macchina.

## Utilizzo

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

### Regole di didascalia per progetto (`caption-rules.json`) e provenienza

Posiziona `caption-rules.json` in `<project>/caption-rules.json` (o passa `--caption-config path/to/rules.json`):

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

- **Controlli del gate (senza modello, riportati in `description-review.jsonl`):** `prefix_missing` (la didascalia non inizia con `prefix`), `required_term_missing` (manca una voce di `required_terms` per la classe del record — risolta da `class_field` o `class_pattern` — oppure uno sfondo `required`), `banned_word` (compare una frase di `banned_words` globale, di sfondo o di classe), più `exact_duplicate` e `near_duplicate` a livello di pool (Jaccard ≥ 0.85).
- **Composizione (`compose` o `report --compose`):** scrive `<out>/captions-manifest.json` (`--manifest-name` sostituisce il nome del file) nel formato lane-3 `run_dataset` di `style-dataset-lab` `{"items":[{"image","sha256","caption",...}]}` — mai all'interno di `--project`. Ogni didascalia viene assemblata da `prefix`, dai termini obbligatori di `class` riportati alla lettera, da un `backdrop` facoltativo e dalla parte libera di contenuto visibile (la modifica `seat` quando importata, altrimenti il testo umano/sidecar approvato — mai una bozza non modificata). Per impostazione predefinita, `compose` e `report --compose` redigono bozze **solo** dei record contrassegnati dal gate senza modello, nella coda di revisione; `--all` scrive anche proposte facoltative `draft_offer` / `draft_offer_writer` accanto agli originali non contrassegnati, senza sostituire il testo umano o sidecar.
- **Registro di provenienza:** ogni didascalia finale, sia in `captions-manifest.json` sia in `description-review.jsonl`, porta un elenco `writers` per parte che usa `config:<file>:<line>`, `drafter:<id>@<rev>:<prompt sha>`, `seat:<name>`, `record:<file>` e `sidecar:<file>`, con lo strato `captioner_assisted` quando è stata modificata da un seat dopo aver visto una bozza o una proposta, `"human"` per i record umani e `"sidecar"` per i sidecar `.txt`.
- **Importazione dei sidecar (`sidecar-import` o `--sidecars`):** abbina ogni immagine al suo sidecar `.txt` con lo stesso nome base, assegna `author_kind: "sidecar"` (mai `"human"`), estrae i prompt di generazione ComfyUI incorporati nei PNG quando presenti, esegue il gate completo delle descrizioni (eco del prompt, duplicati esatti/Jaccard e `caption-rules.json`), segnala ogni immagine senza sidecar `.txt` invece di inventare testo e scrive tutti i record importati e i file di revisione rigorosamente sotto `--out`.
- **Blocco del pool per tag, protezione fail-closed contro le copie rinominate e stato `pool_detection`:** `check_multimodal_pool` applica il blocco del benchmark sia a `--project` sia ai comandi di testo (`text-index`, `text-dupes`, `text-search`, `lineage`) quando il nome della cartella `realpath` oppure `"multimodal_pool"` nella configurazione (`caption-rules.json` o `project.json`) indica un pool bloccato. Per le cartelle senza tag al di fuori di quei nomi, calcola l'hash dell'insieme esatto di immagini lette dalla scansione (inclusi collegamenti simbolici e giunzioni di directory di Windows) e lo confronta con un manifest locale non vuoto (`DATASET_DESK_POOL_MANIFEST` o `pool_manifest` in `~/.config/dataset-desk/config.json`, tenuto fuori dal repository) oppure con `--locked-pool-dir <path>` — e, nei comandi di testo, controlla anche i nomi base dei sidecar di didascalia `.txt` rispetto a quelli delle immagini del pool bloccato — rifiutando in modalità fail-closed (`PERM_LOCKED_POOL`) con l'indicazione del pool corrispondente e del conteggio. Ogni comando risolve un unico stato `pool_detection` (`status: "on"` con >= 1 hash di immagine per ogni pool configurato e conteggi per sorgente — mai valori di hash — oppure `status: "off"` con il testo di avviso, emesso una sola volta su `stderr`) e registra `pool_detection` in `<out>/pool-detection.json`, negli output JSON/JSONL sotto `--out` e in una nota di una riga in ogni report Markdown.
  - Il rilevamento delle copie rinominate confronta le immagini tramite `sha256`, quindi né una copia composta solo da immagini nuove (nessuna condivisa con un pool bloccato) né una copia ridimensionata, ricodificata o priva di metadati corrisponde tramite `sha256`; contrassegna tali progetti con il tag `"multimodal_pool"` in `caption-rules.json` o `project.json`.

`dataset-desk --help` elenca tutti i flag. Codici di uscita: `0` ok, `1` un errore dell'utente o un rifiuto, `2` un errore imprevisto. Gli errori vengono stampati come `error <CODE>: <message>` con una riga `hint:`; `--debug` aggiunge la traccia dello stack.

**Il layout del progetto che legge** è quello di [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un progetto senza `records/` funziona comunque, a condizione che le immagini si trovino in cartelle sotto `outputs/` (i file nella parte superiore di `outputs/` vengono ignorati come fogli di contatto): ogni immagine è "non registrata" e i controlli di duplicati e di aspetto vengono comunque eseguiti.

## Modello di minaccia e fiducia

- **Cosa legge:** la cartella del progetto che si specifica con `--project`: immagini, `records/*.json`, metadati wave e i dati di generazione incorporati nei file PNG. I comandi di testo leggono la cartella `--source` o il checkout `--rnd-data` specificati. Rifiutano l'archivio privato e i blob di rnd-data e stampano solo i percorsi relativi.
- **Cosa scrive:** solo sotto `--out`: la cache di scansione, le cache di incorporamento, il report, `description-review.jsonl` e il foglio di lavoro. Rifiuta un `--out` all'interno del progetto (`PERM_OUT_INSIDE_PROJECT`), quindi un progetto non viene mai modificato.
- **Cosa non tocca mai:** qualsiasi cosa al di fuori di queste due cartelle. Non elimina nulla, da nessuna parte.
- **Rete:** la versione lite non la utilizza mai. Un backend del modello recupera i suoi pesi fissati da Hugging Face al primo utilizzo, a meno che `HF_HUB_OFFLINE=1` non sia impostato; il generatore viene caricato con `HF_HUB_OFFLINE=1` e `local_files_only=True` ogni volta. Nessun modello esegue codice remoto.
- **Autorizzazioni:** un account utente ordinario. Una GPU richiede `--device` più un ID `--grant`, in modo che una scheda condivisa non venga utilizzata per errore. Prima di caricare un modello, un regolatore VRAM verifica che la memoria già in uso più il fabbisogno misurato del modello rientrino in `--vram-budget` (impostazione predefinita 26.624 MiB). Non rimuove mai nulla; `--wait` esegue il polling finché il caricamento non rientra nei limiti.
- **Segreti e telemetria:** non legge credenziali e non raccoglie né invia telemetria.
- **Passaggi protetti:** nei progetti bloccati per un benchmark in esecuzione, gli incorporamenti di descrizione e qualsiasi accettazione delle descrizioni basata sul modello vengono rifiutati a meno che non venga passato `--allow-multimodal-pool`, in modo che le query del benchmark non siano influenzate dai modelli che misura.

Segnala una vulnerabilità come descritto in [SECURITY.md](SECURITY.md).

## Stato

**v0.3.0, un prototipo pubblico.** Il codice viene testato offline con piccoli esempi che simulano le librerie del modello.
- **Calibrazione:** il sistema viene eseguito con i propri dati, su una RTX 5090, e vengono impostate le soglie per i duplicati, che sono provvisorie e si basano su un singolo pool per tipo. I valori di VRAM sono valori conservativi relativi all'intera scheda, incluso il contesto CUDA, misurati su una RTX 5090; altre schede e corpus potrebbero differire. Le soglie del codice si basano su 17 copie inserite. Il metodo e i dati sono disponibili in [newsletter, numero 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Progetti di grandi dimensioni:** i duplicati, i valori anomali e il canale di ricerca funzionano in blocchi delimitati. Su set di dati sintetici, la ricerca di duplicati e il canale di ricerca richiedono pochi minuti per 100.000 immagini, mentre prima richiedevano ore o esaurivano la memoria. Il controllo dei duplicati del filtro di descrizione utilizza un filtro di prefisso esatto, quindi confronta solo le descrizioni che potrebbero raggiungere 0,85. La prima scansione è circa 2 volte più veloce per immagine, con risultati identici, e viene eseguita in parallelo sotto `--threads`.
- **È previsto che la riga di comando cambi prima della versione 1.0.**

La documentazione di progettazione, le basi della ricerca e la cronologia delle revisioni sono disponibili nella libreria di ricerca del sistema: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Sviluppo

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

Il sistema di integrazione continua (CI) mantiene la copertura del codice al 90% o superiore.

## Licenza

[MIT](LICENSE). I modelli che può caricare non fanno parte di questo pacchetto e ciascuno ha la propria licenza; leggere la scheda del modello prima di utilizzarlo.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
