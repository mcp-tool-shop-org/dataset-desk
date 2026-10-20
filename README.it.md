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

Uno stile LoRA apprende dalle immagini di addestramento e dalle relative didascalie. Due difetti minori ne compromettono il funzionamento:
- **Ripetizioni.** La stessa immagine salvata due volte, o due rendering leggermente diversi, fanno sì che la stessa immagine venga "vista" due volte.
- **Descrizioni che non sono descrizioni.** Una didascalia copiata dal prompt di generazione, o una regola applicata a 54 immagini, non fornisce al modello alcuna informazione sull'immagine. Peggio ancora, se questo testo viene successivamente utilizzato per testare la ricerca, misura la capacità di recupero del prompt, e non la comprensione dell'immagine.

dataset-desk individua entrambi i problemi, all'interno di una cartella di progetto, senza modificarla. Genera un report e una coda di revisione per l'intervento di un operatore. **Niente viene rimosso automaticamente.**

## Cosa fa

| Comando | Risultato |
|---|---|
| `index` | Scansiona il progetto (record, metadati audio, dati di generazione PNG) e memorizza ogni immagine univoca in una cache all'interno di `--out` |
| `dupes` | Copie identiche (sha256), duplicati quasi identici (dHash a 256 bit entro 10 bit) e duplicati semantici (similarità coseno dell'embedding), in formato JSON |
| `search` | Le immagini più simili a una descrizione (`--text`), a un'immagine (`--image`) o a entrambe |
| `coverage` | Le immagini migliori per ciascun soggetto in un elenco di soggetti desiderati, in modo che le lacune siano evidenti |
| `report` | Un report in formato Markdown: duplicati, le immagini approvate meno tipiche in base al contenuto e all'aspetto, la verifica delle descrizioni e se ciascuna descrizione corrisponde alla propria immagine |

Il **filtro delle descrizioni** funziona senza alcun modello. Una descrizione viene segnalata quando:
- ripete il prompt: è uguale al prompt o è contenuto in esso, condivide una sequenza di 6 o più parole con esso, o il 90% o più delle sue parole (80% o più delle parole rilevanti) compaiono in esso;
- include la riga dell'oggetto del record: una sequenza di 4 o più parole `canon.subject` compare come una sequenza continua di parole;
- è troppo breve: meno di 8 parole o meno di 5 parole rilevanti;
- duplica un'altra descrizione nell'insieme: identica dopo aver rimosso la punteggiatura, o un indice di Jaccard delle parole rilevanti pari a 0,85 o superiore;
- è troppo lunga per SigLIP2: più di 64 token (l'unico controllo che richiede un tokenizer; i record non controllati non vengono mai contrassegnati come validi).

I record segnalati vengono inviati a `description-review.jsonl` e a una **scheda di lavoro anonima**: ogni immagine segnalata è affiancata da 3-5 immagini visivamente più simili, senza prompt, senza riga dell'oggetto e senza il testo precedente, in modo che l'operatore che la modifica possa descrivere l'immagine. Un redattore opzionale ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) può scrivere una prima bozza basandosi solo sui pixel; le bozze vengono registrate come `captioner_assisted` e un operatore le modifica.

Il **canale visivo** è costituito da 30 numeri per immagine (palette, tonalità e texture alla risoluzione nativa), in modo che "il rendering di questa immagine è errato" sia distinto da "il soggetto di questa immagine è strano".

## Installazione

```bash
pip install dataset-desk
```

Questo fornisce il **livello base**: solo numpy e Pillow, senza torch. I controlli sui duplicati, il canale visivo, il filtro delle descrizioni e il report funzionano in questo livello (`--backend none`).

Per l'indice di embedding (ricerca, copertura, duplicati semantici, controllo della corrispondenza della descrizione), aggiungere le librerie del modello:

```bash
pip install "dataset-desk[models]"
```

I modelli vengono scaricati da Hugging Face con le revisioni specificate la prima volta che vengono utilizzati, quindi vengono caricati dalla cache: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`--backend eg2`) o [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) (`--backend siglip2`). Impostare `HF_HUB_OFFLINE=1` per impedire i download. Il redattore plain-sight è sempre offline: non scarica mai nulla.

**Funziona su:** Python 3.11 e 3.12, su Windows e Linux. CPU per impostazione predefinita, con ogni pool di thread limitato (4 thread a meno che `--threads` non indichi diversamente) e l'esecuzione di un modello riduce la priorità del processo a sotto la norma, in modo che possa condividere una macchina.

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
```

`dataset-desk --help` elenca tutti i flag. Codici di uscita: `0` ok, `1` un errore dell'utente o un rifiuto, `2` un errore imprevisto. Gli errori vengono visualizzati come `error <CODE>: <message>` con una riga `hint:`; `--debug` aggiunge la traccia dello stack.

Il **layout del progetto che legge** è quello di [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un progetto senza `records/` funziona comunque, a condizione che le immagini si trovino in cartelle all'interno di `outputs/` (i file nella parte superiore di `outputs/` vengono ignorati come fogli di contatto): ogni immagine è "non registrata" e i controlli sui duplicati e sull'aspetto continuano a essere eseguiti.

## Modello di minaccia e affidabilità

- **Cosa legge:** la cartella del progetto specificata con `--project`: immagini, `records/*.json`, metadati audio e i dati di generazione incorporati nei file PNG.
- **Cosa scrive:** solo all'interno di `--out`: la cache di scansione, le cache di embedding, il report, `description-review.jsonl` e la scheda di lavoro. Rifiuta un `--out` all'interno del progetto (`PERM_OUT_INSIDE_PROJECT`), quindi un progetto non viene mai modificato.
- **Cosa non tocca mai:** qualsiasi cosa al di fuori di queste due cartelle. Non elimina nulla, da nessuna parte.
- **Rete:** il livello base non la utilizza mai. Un backend del modello scarica i suoi pesi fissati da Hugging Face alla prima esecuzione, a meno che `HF_HUB_OFFLINE=1` non sia impostato; il redattore viene caricato con `HF_HUB_OFFLINE=1` e `local_files_only=True` ogni volta. Nessun modello esegue codice remoto.
- **Autorizzazioni:** un account utente ordinario. Una GPU richiede `--device` più un ID `--grant`, in modo che una scheda condivisa non venga utilizzata accidentalmente.
- **Segreti e telemetria:** non legge credenziali e non raccoglie né invia telemetria.
- **Passaggi protetti:** nei progetti bloccati per un benchmark in esecuzione, l'embedding delle descrizioni e qualsiasi accettazione delle descrizioni basata su modelli vengono rifiutati a meno che `--allow-multimodal-pool` non venga passato, in modo che le query del benchmark non siano influenzate dai modelli che misura.

Segnala una vulnerabilità come descritto in [SECURITY.md](SECURITY.md).

## Stato

**v0.1.3, un prototipo pubblico.** Il codice è stato testato offline (152 test, 96% di copertura delle righe, con piccoli sostituti per le librerie del modello) e una prima calibrazione sui dati dello studio ha classificato i modelli in base all'attività, ma non ha ancora fissato le soglie: le soglie per i duplicati quasi identici e i duplicati semantici sono ancora solo punti di partenza. Prevedi che la riga di comando cambierà prima della versione 1.0.

La documentazione del progetto, le basi della ricerca e la cronologia delle revisioni sono conservate nella biblioteca di ricerca dello studio: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Sviluppo

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

Il sistema di integrazione continua (CI) garantisce una copertura del codice pari o superiore al 90%.

## Licenza

[MIT](LICENSE). I modelli che può caricare non fanno parte di questo pacchetto e ciascuno è soggetto alla propria licenza; leggere la scheda del modello prima di utilizzarlo.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
