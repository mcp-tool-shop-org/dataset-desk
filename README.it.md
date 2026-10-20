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

Uno stile LoRA apprende dalle immagini di addestramento e dalle relative didascalie. Due difetti minori ne compromettono l'efficacia:
- **Ripetizioni.** La stessa immagine salvata due volte, o due rendering leggermente diversi, fanno sì che la stessa immagine venga "vista" due volte.
- **Descrizioni che non sono descrizioni.** Una didascalia copiata dal prompt di generazione, o una regola applicata a 54 immagini, non fornisce al modello alcuna informazione sull'immagine. Peggio ancora, se questo testo viene successivamente utilizzato per testare la ricerca, misura la capacità di recupero del prompt, non la comprensione dell'immagine.

dataset-desk individua entrambi i problemi all'interno di una cartella di progetto, senza modificarla. Genera un report e una coda di revisione per l'intervento di un operatore. **Niente viene rimosso automaticamente.**

## Cosa fa

| Comando | Risultato |
|---|---|
| `index` | Scansiona il progetto (record, metadati delle onde, dati di generazione PNG) e inserisce ogni immagine univoca in una cache all'interno di `--out` |
| `dupes` | Copie identiche (sha256), duplicati quasi identici (dHash a 256 bit con consapevolezza della simmetria) e duplicati semantici (similarità coseno dell'embedding), in formato JSON. Utilizza un indice DINOv2 e applica una regola a due segnali calibrata. Due viste dello stesso oggetto non vengono mai accoppiate. |
| `search` | Le immagini più simili a una descrizione (`--text`), a un'immagine (`--image`) o a entrambe. |
| `coverage` | Le immagini migliori per ciascun soggetto in un elenco di soggetti desiderati, in modo che le lacune siano evidenti. |
| `report` | Un report in formato Markdown: duplicati, le immagini approvate meno tipiche in base al contenuto e all'aspetto, l'analisi delle descrizioni e se ciascuna descrizione corrisponde alla propria immagine. |
| `text-index` | Inserisce i file di testo e codice di una cartella `--source` (o di un checkout rnd-data), contando i blocchi utilizzando il tokenizer del modello. |
| `text-dupes` | File identici e quasi duplicati. `--task code` confronta solo gli script. |
| `text-search` | I file più simili a una query (`--text`). `--task code` utilizza il prompt di recupero del codice del modello. |
| `lineage` | Versioni dello stesso script tra le diverse esecuzioni, ordinate per similarità. |

**Le soglie** derivano dalla calibrazione del software sui propri dati. Ognuna di esse è contrassegnata come provvisoria nel codice, con l'indicazione della misurazione da cui deriva:
- **Immagini:** su un indice DINOv2, una coppia è considerata un duplicato quando `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Altri indici utilizzano il dHash entro 26 bit, più la similarità coseno ≥ 0,95.
- **Testo:** 0,98 con bge-m3 e 0,95 per il codice con EmbeddingGemma 2. Qualsiasi altro modello richiede `--semantic`.

Due rendering dello stesso oggetto da angolazioni diverse non sono mai considerati duplicati. La vista deriva da `provenance.asset_id` e `render_id` di ciascun record, o dai nomi dei file come `<asset>__y<yaw>_e<elev>`.

**Il filtro delle descrizioni** viene eseguito senza l'utilizzo di alcun modello. Una descrizione viene segnalata quando:
- fa eco al suo prompt: è uguale al prompt o è contenuto in esso, condivide una sequenza di 6 o più parole con esso, o il 90% o più delle sue parole (80% o più delle sue parole chiave) compaiono in esso;
- racchiude la riga dell'oggetto del record: una sequenza di 4 o più parole `canon.subject` compare come una sequenza contigua delle sue parole;
- è troppo breve: meno di 8 parole o meno di 5 parole chiave;
- duplica un'altra descrizione nell'insieme: identica dopo aver rimosso la punteggiatura, o con un indice di Jaccard delle parole chiave pari a 0,85 o superiore;
- è troppo lunga per SigLIP2: più di 64 token (l'unico controllo che richiede un tokenizer; i record non controllati non vengono mai contrassegnati come validi).

I record segnalati vengono inviati a `description-review.jsonl` e a una **scheda di lavoro anonima**: ogni immagine segnalata è affiancata da 3-5 immagini visivamente più simili, senza prompt, senza riga dell'oggetto e senza il vecchio testo in vista, in modo che la persona che la riscrive descriva l'immagine. Un redattore opzionale ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) può scrivere una prima bozza a partire dai soli pixel; le bozze vengono registrate come `captioner_assisted` e una persona le modifica.

**Il canale dell'aspetto** è costituito da 30 numeri per immagine (palette, tonalità e texture alla risoluzione nativa), in modo che "il rendering di questa immagine è errato" sia distinto da "il soggetto di questa immagine è strano".

## Installazione

```bash
pip install dataset-desk
```

Questo fornisce la **versione "lite"**: solo numpy e Pillow, senza torch. I controlli sui duplicati, il canale dell'aspetto, il filtro delle descrizioni e il report funzionano tutti in questa versione (`--backend none`).

Per l'indice di embedding (ricerca, copertura, duplicati semantici, controllo della corrispondenza della descrizione), aggiungere le librerie del modello:

```bash
pip install "dataset-desk[models]"
```

Su Windows con una GPU NVIDIA, installare prima una versione CUDA di torch da `https://download.pytorch.org`; la versione di torch di PyPI per Windows è solo CPU.

I modelli vengono scaricati da Hugging Face alla prima esecuzione, alle revisioni specificate, e quindi vengono caricati dalla cache. Ogni attività ha un valore predefinito e `--backend` ne seleziona un altro:

| Backend | Modello | Predefinito per |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | testo |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | codice (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

EmbeddingGemma 2 viene utilizzato nel modo indicato nella sua scheda del modello: un prefisso di attività per ogni comando, caricamento solo di testo per il testo e `--dim 768|512|256`. Impostare `HF_HUB_OFFLINE=1` per impedire i download. Il redattore plain-sight è sempre offline: non effettua mai download.

**Funziona su:** Python 3.11 e 3.12, su Windows e Linux. CPU per impostazione predefinita, con ogni pool di thread limitato (4 thread a meno che `--threads` non indichi diversamente) e un'esecuzione del modello riduce la priorità del processo a sotto la norma, in modo che possa condividere una macchina.

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
```

`dataset-desk --help` elenca tutti i flag. Codici di uscita: `0` ok, `1` un errore dell'utente o un rifiuto, `2` un errore imprevisto. Gli errori vengono stampati come `error <CODE>: <message>` con una riga `hint:`; `--debug` aggiunge la traccia dello stack.

**Il layout del progetto che legge** è quello di [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un progetto senza `records/` funziona comunque, a condizione che le immagini si trovino in cartelle all'interno di `outputs/` (i file nella parte superiore di `outputs/` vengono ignorati come fogli di contatto): ogni immagine è "non registrata" e i controlli sui duplicati e sull'aspetto vengono comunque eseguiti.

## Modello di minaccia e fiducia

- **Cosa legge:** la cartella del progetto a cui assegna il nome `--project`: immagini, `records/*.json`, metadati delle onde e i dati di generazione incorporati nei file PNG. I comandi di testo leggono la cartella `--source` o eseguono il checkout dalla cartella `--rnd-data` a cui assegna il nome. Rifiutano l'archivio privato e i blob di rnd-data e stampano solo i percorsi relativi.
- **Cosa scrive:** solo in `--out`: la sua cache di scansione, le cache di incorporamento, il report, `description-review.jsonl` e il foglio di calcolo. Rifiuta un `--out` all'interno del progetto (`PERM_OUT_INSIDE_PROJECT`), quindi un progetto non viene mai modificato.
- **Cosa non tocca mai:** qualsiasi cosa al di fuori di queste due cartelle. Non elimina nulla, da nessuna parte.
- **Rete:** la versione lite non la utilizza mai. Un backend del modello recupera i suoi pesi fissati da Hugging Face al primo utilizzo, a meno che non sia impostato `HF_HUB_OFFLINE=1`; il modulo di generazione viene caricato con `HF_HUB_OFFLINE=1` e `local_files_only=True` ogni volta. Nessun modello esegue codice remoto.
- **Autorizzazioni:** un account utente ordinario. Una GPU necessita di `--device` più un ID `--grant`, in modo che una scheda condivisa non venga utilizzata per errore. Prima di caricare un modello, un gestore della VRAM verifica che la memoria già in uso più il fabbisogno misurato del modello rientrino in `--vram-budget` (valore predefinito 26.624 MiB). Non elimina mai nulla; `--wait` esegue il polling finché il caricamento non è completo.
- **Segreti e telemetria:** non legge credenziali e non raccoglie né invia telemetria.
- **Passaggi protetti:** per i progetti bloccati per un benchmark in esecuzione, gli incorporamenti delle descrizioni e qualsiasi accettazione delle descrizioni basata su un modello vengono rifiutati a meno che non venga passato `--allow-multimodal-pool`, in modo che le query del benchmark non siano influenzate dai modelli che misura.

Segnala una vulnerabilità come descritto in [SECURITY.md](SECURITY.md).

## Stato

**v0.2.2, un prototipo pubblico.** Il codice viene testato offline: 197 test e il 96% delle righe coperte, con piccole sostituzioni per le librerie del modello.
- **Calibrazione:** le esecuzioni dello studio sui propri dati, su una RTX 5090, hanno impostato le soglie di duplicazione, che sono provvisorie e si basano su un pool per tipo. Le cifre della VRAM sono cifre conservative per l'intera scheda, incluso il contesto CUDA, misurate su una RTX 5090; altre schede e corpora potrebbero differire. Le soglie del codice si basano su 17 copie inserite. Il metodo e i dati sono disponibili in [newsletter issue 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Progetti di grandi dimensioni:** i duplicati, i valori anomali e il canale di ricerca funzionano in blocchi delimitati. Sui set sintetici, la ricerca di duplicati e il canale di ricerca richiedono pochi minuti con 100.000 immagini, mentre prima richiedevano ore o esaurivano la memoria. Il controllo dei duplicati quasi identici del gateway di descrizione utilizza un filtro di prefisso esatto, quindi confronta solo le descrizioni che potrebbero raggiungere 0,85. La prima scansione è circa 2 volte più veloce per immagine, con risultati identici, e viene eseguita in parallelo in `--threads`.
- **È previsto che la riga di comando cambi prima della versione 1.0.**

Il documento di progettazione, le sue basi di ricerca e la cronologia delle revisioni sono disponibili nella libreria di ricerca dello studio: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Sviluppo

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI mantiene la copertura delle righe al 90% o superiore.

## Licenza

[MIT](LICENSE). I modelli che può caricare non fanno parte di questo pacchetto e ciascuno ha la propria licenza; leggi la scheda del modello prima di utilizzarne uno.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
