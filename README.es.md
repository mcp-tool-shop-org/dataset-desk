<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

## ¿Por qué?

Un estilo LoRA aprende de sus imágenes de entrenamiento y de sus descripciones. Dos defectos sutiles lo arruinan:
- **Repeticiones.** La misma imagen guardada dos veces, o dos renderizados que difieren en unos pocos píxeles, enseñan la misma imagen dos veces.
- **Descripciones que no son descripciones.** Una descripción copiada del mensaje de generación, o una regla pegada en 54 imágenes, no le dice nada al modelo sobre la imagen. Peor aún, si ese texto se utiliza más tarde para probar la búsqueda, mide la recuperación del mensaje, no la comprensión de la imagen.

dataset-desk detecta ambos, en una carpeta de proyecto, sin modificarla. Escribe un informe y una cola de revisión para que una persona actúe sobre ellos. **Nada se elimina automáticamente.**

## Qué hace

| Comando | Qué se obtiene |
|---|---|
| `scan` | Escanea el proyecto en `scan.jsonl` y `scan-report.json` bajo `--out` sin cargar un codificador. |
| `index` | Escanea el proyecto (registros, metadatos de onda, datos de generación PNG) y codifica cada imagen única en una caché bajo `--out`. |
| `dupes` | Copias idénticas (sha256), duplicados cercanos (un dHash de 256 bits con conocimiento de reflejos) y duplicados semánticos (coseno de incrustación), en formato JSON. En un índice DINOv2, utiliza una regla calibrada de dos señales. Dos vistas del mismo objeto nunca se emparejan. |
| `search` | Las imágenes más cercanas a una descripción (`--text`), una imagen (`--image`) o ambas juntas. |
| `coverage` | Las mejores imágenes para cada sujeto en una lista de sujetos deseados, para que se muestren los huecos. |
| `report` | Un informe en Markdown: duplicados, las imágenes aprobadas menos típicas por contenido y por apariencia, la auditoría de la descripción y si cada descripción encuentra su propia imagen. |
| `sidecar-import` | Importa carpetas de imágenes emparejadas con archivos adjuntos de descripción con el mismo prefijo `.txt` (`--sidecars` en `index`/`report`/`compose` hace lo mismo), ejecuta la puerta de la descripción, informa sobre las imágenes sin un archivo adjunto `.txt` sin inventar texto y escribe todo bajo `--out`. |
| `compose` | Compone las descripciones finales de entrenamiento en `--out` a partir de reglas por proyecto y ediciones importadas (por defecto, solo se incluyen en la cola de revisión los registros marcados, o se ofrecen borradores lado a lado con `--all`, sin reemplazar el texto humano o el archivo adjunto), y escribe un manifiesto de conjunto de datos de nivel 3 (`{"items":[{"image","sha256","caption"}]}`) con el origen de cada parte. |
| `import-edits` | Reimporta las filas editadas desde `description-worksheet.md` a `--out/seat-edits.jsonl` con un ID de escritor `seat:<name>`. |
| `text-index` | Codifica los archivos de texto y código de una carpeta `--source` (o una copia de rnd-data), contando los fragmentos con el tokenizador del modelo. |
| `text-dupes` | Archivos exactos y casi duplicados. `--task code` compara solo los scripts. |
| `text-search` | Los archivos más cercanos a una consulta (`--text`). `--task code` utiliza el mensaje de recuperación de código del modelo. |
| `lineage` | Versiones del mismo script en diferentes ejecuciones, clasificadas por similitud. |

**Los umbrales** provienen de la calibración del estudio con sus propios datos. Cada uno está marcado como provisional en el código, con la medida de la que proviene:
- **Imágenes:** en un índice DINOv2, un par es un duplicado cuando `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Otros índices utilizan el dHash dentro de 26 bits, más un coseno ≥ 0,95.
- **Texto:** 0,98 con bge-m3 y 0,95 para código con EmbeddingGemma 2. Cualquier otro modelo necesita `--semantic`.

Dos renderizados del mismo objeto desde diferentes ángulos de cámara nunca son duplicados. La vista proviene del `provenance.asset_id` y el `render_id` de cada registro, o de nombres de archivo como `<asset>__y<yaw>_e<elev>`.

**La puerta de la descripción** se ejecuta sin ningún modelo. Se marca una descripción cuando:
- hace eco de su mensaje: es igual al mensaje o está dentro de él, comparte una secuencia de 6 o más palabras con él, o el 90 % o más de sus palabras (el 80 % o más de sus palabras de contenido) aparecen en él;
- envuelve la línea de asunto del registro: una secuencia de 4 o más palabras `canon.subject` aparece como una secuencia continua de sus palabras;
- es demasiado corta: menos de 8 palabras o menos de 5 palabras de contenido;
- duplica otra descripción en el conjunto: idéntica una vez que se elimina la puntuación, o un Jaccard de palabras de contenido de 0,85 o más;
- es demasiado larga para SigLIP2: más de 64 tokens (la única comprobación que necesita un tokenizador; los registros no comprobados nunca se marcan como válidos);
- viola las reglas por proyecto `caption-rules.json` (cuando están presentes): `prefix_missing`, `required_term_missing` o `banned_word` (junto con las comprobaciones de la agrupación `exact_duplicate` y `near_duplicate`).

Los registros marcados se envían a `description-review.jsonl` y a una **hoja de trabajo ciega**: cada imagen marcada junto con 3 a 5 hermanos visualmente más cercanos (elegidos de la similitud solo de imágenes DINOv2 cuando se almacenan en caché bajo `--out`, de lo contrario dHash; nunca incrustaciones de texto en un grupo bloqueado), sin ningún mensaje, sin ninguna línea de asunto y sin ningún texto antiguo a la vista, para que la persona que lo reescriba describa la imagen. Un redactor opcional (`--draft plain-sight` por defecto, o `--draft qwen3-vl`) puede escribir un primer borrador a partir de los píxeles en la cola de revisión para el equipo (nunca directamente en la descripción de entrenamiento); se guarda el resultado completo del redactor por imagen en `drafter-raw.jsonl`. Pasar `--all` también incluye los registros no marcados como ofertas lado a lado (`draft_offer` / `draft_offer_writer`) sin reemplazar el texto humano o el archivo adjunto. Los borradores editados por el equipo se registran como `captioner_assisted`.

**El canal de apariencia** son 30 números por imagen (paleta, tono y textura en resolución nativa), por lo que "el renderizado de esta imagen es incorrecto" se muestra por separado de "el sujeto de esta imagen es extraño".

## Instalación

```bash
pip install dataset-desk
```

Eso proporciona el **nivel básico**: solo numpy y Pillow, sin torch. Las comprobaciones de duplicados, el canal de apariencia, la puerta de la descripción y el informe funcionan en él (`--backend none`).

Para el índice de incrustación (búsqueda, cobertura, duplicados semánticos, la comprobación de ajuste de la descripción), agregue las bibliotecas del modelo:

```bash
pip install "dataset-desk[models]"
```

En Windows con una GPU NVIDIA, instale primero una versión de torch compilada con CUDA desde `https://download.pytorch.org`; la versión de torch de PyPI para Windows es solo para CPU.

Los modelos se obtienen de Hugging Face en las versiones fijas la primera vez que se utilizan y, a continuación, se cargan desde su caché. Cada tarea tiene un valor predeterminado, y `--backend` elige otro:

| Backend | Modelo | Predeterminado para |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | texto |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | código (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

Redactores de imágenes ciegos (`--draft`, `dataset_desk.registry.DRAFTER_REGISTRY`; ambos declaran `is_arm_model=False`, por lo que los grupos de referencia bloqueados les permiten redactar):

| Redactor | Modelo | Fijar | Licencia | Predeterminado |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | sí |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | — |

EmbeddingGemma 2 se utiliza de la manera que indica su tarjeta de modelo: un prefijo de tarea por comando, carga solo de texto para texto y `--dim 768|512|256`. Establezca `HF_HUB_OFFLINE=1` para prohibir las descargas. `plain-sight` siempre está desconectado; `qwen3-vl` respeta `HF_HUB_OFFLINE` (`local_files_only=True`), rechaza cualquier discrepancia de revisión (`MODEL_REVISION_MISMATCH`), respeta el gobernador de VRAM, acepta un indicador por proyecto (`drafter_prompt` o `--drafter-prompt`) y almacena el indicador y la salida completa sin procesar por imagen.

**Funciona en:** Python 3.11 y 3.12, en Windows y Linux. CPU de forma predeterminada, con cada grupo de subprocesos limitado (4 subprocesos a menos que `--threads` indique lo contrario), y la ejecución de un modelo reduce el proceso a una prioridad inferior a la normal, para que pueda compartir una máquina.

## Uso

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

### Reglas de subtítulos por proyecto (`caption-rules.json`) y procedencia

Coloque `caption-rules.json` en `<project>/caption-rules.json` (o pase `--caption-config path/to/rules.json`):

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

- **Comprobaciones de la puerta (sin modelo, informadas en `description-review.jsonl`):** `prefix_missing` (la descripción no empieza por `prefix`), `required_term_missing` (falta alguna entrada de `required_terms` para la clase del registro — resuelta a partir de `class_field` o `class_pattern` — o un fondo `required`), `banned_word` (aparece alguna frase de `banned_words` global, de fondo o de clase), además de `exact_duplicate` y `near_duplicate` del conjunto (Jaccard ≥ 0.85).
- **Composición (`compose` o `report --compose`):** escribe `<out>/captions-manifest.json` (`--manifest-name` sobrescribe el nombre del archivo) en el formato lane-3 `run_dataset` de `style-dataset-lab` `{"items":[{"image","sha256","caption",...}]}` — nunca dentro de `--project`. Cada descripción se ensambla a partir de `prefix`, los términos obligatorios de `class` tal cual, un `backdrop` opcional y la parte libre de contenido visible (la edición `seat` cuando se ha importado; en caso contrario, el texto humano/sidecar aprobado — nunca un borrador sin editar). Por defecto, `compose` y `report --compose` redactan borradores **solo** de los registros marcados por la puerta sin modelo, en la cola de revisión; `--all` también escribe propuestas opcionales `draft_offer` / `draft_offer_writer` junto a los originales sin marcar, sin reemplazar el texto humano o sidecar.
- **Registro de procedencia:** cada descripción final, tanto en `captions-manifest.json` como en `description-review.jsonl`, lleva una lista `writers` por parte que usa `config:<file>:<line>`, `drafter:<id>@<rev>:<prompt sha>`, `seat:<name>`, `record:<file>` y `sidecar:<file>`, con el estrato `captioner_assisted` cuando un seat la editó tras ver un borrador o una propuesta, `"human"` para los registros humanos y `"sidecar"` para los archivos sidecar `.txt`.
- **Importación de sidecar (`sidecar-import` o `--sidecars`):** empareja cada imagen con su archivo sidecar `.txt` del mismo nombre base, asigna `author_kind: "sidecar"` (nunca `"human"`), extrae los prompts de generación de ComfyUI incrustados en los PNG cuando existen, ejecuta la puerta de descripción completa (eco del prompt, duplicados exactos/Jaccard y `caption-rules.json`), informa de cualquier imagen sin sidecar `.txt` en lugar de inventar texto y escribe todos los registros importados y los archivos de revisión estrictamente bajo `--out`.
- **Bloqueo del conjunto por etiqueta, protección fail-closed contra copias renombradas y estado `pool_detection`:** `check_multimodal_pool` aplica el bloqueo de referencia tanto a `--project` como a los comandos de texto (`text-index`, `text-dupes`, `text-search`, `lineage`) cuando el nombre de carpeta `realpath` o `"multimodal_pool"` en la configuración (`caption-rules.json` o `project.json`) designa un conjunto bloqueado. Para las carpetas sin etiqueta fuera de esos nombres, calcula el hash del conjunto exacto de imágenes que lee el escaneo (incluidos enlaces simbólicos y uniones de directorio de Windows) y lo compara con un manifiesto local no vacío (`DATASET_DESK_POOL_MANIFEST` o `pool_manifest` en `~/.config/dataset-desk/config.json`, guardado fuera del repositorio) o con `--locked-pool-dir <path>` — y, en los comandos de texto, también comprueba los nombres base de los sidecars de descripción `.txt` frente a los de las imágenes del conjunto bloqueado — y rechaza en modo fail-closed (`PERM_LOCKED_POOL`) indicando el conjunto coincidente y el recuento. Cada comando resuelve un único estado `pool_detection` (`status: "on"` con >= 1 hash de imagen por conjunto configurado y recuentos por fuente — nunca valores de hash — o `status: "off"` con el texto de advertencia, emitido una sola vez en `stderr`) y registra `pool_detection` en `<out>/pool-detection.json`, en las salidas JSON/JSONL bajo `--out` y en una nota de una línea en cada informe Markdown.
  - La detección de copias renombradas compara las imágenes por `sha256`, de modo que ni una copia hecha solo con imágenes nuevas (ninguna compartida con un conjunto bloqueado) ni una copia redimensionada, recodificada o sin metadatos coincide por `sha256`; etiqueta esos proyectos con `"multimodal_pool"` en `caption-rules.json` o `project.json`.

`dataset-desk --help` enumera todas las marcas. Códigos de salida: `0` correcto, `1` un error de usuario o un rechazo, `2` un error inesperado. Los errores se imprimen como `error <CODE>: <message>` con una línea `hint:`; `--debug` agrega el rastreo de la pila.

**El diseño del proyecto que lee** es el de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un proyecto sin `records/` también funciona, siempre que las imágenes estén en carpetas debajo de `outputs/` (los archivos en la parte superior de `outputs/` se omiten como hojas de contacto): cada imagen es "no registrada" y las comprobaciones de duplicados y de apariencia aún se ejecutan.

## Modelo de amenazas y confianza

- **Lo que lee:** la carpeta del proyecto que nombra con `--project`: imágenes, `records/*.json`, metadatos de onda y los datos de generación incrustados en los archivos PNG. Los comandos de texto leen la carpeta `--source` o el checkout `--rnd-data` que nombra. Rechazan el archivo adjunto privado y los almacenes de datos de rnd-data, y solo imprimen rutas relativas.
- **Lo que escribe:** solo en `--out`: su caché de escaneo, cachés de incrustación, el informe, `description-review.jsonl` y la hoja de cálculo. Rechaza un `--out` dentro del proyecto (`PERM_OUT_INSIDE_PROJECT`), por lo que nunca se modifica un proyecto.
- **Lo que nunca toca:** nada fuera de esas dos carpetas. No elimina nada, en ninguna parte.
- **Red:** la capa lite nunca la usa. Un backend de modelo obtiene sus pesos fijados de Hugging Face en el primer uso, a menos que se establezca `HF_HUB_OFFLINE=1`; el generador se carga con `HF_HUB_OFFLINE=1` y `local_files_only=True` cada vez. Ningún modelo ejecuta código remoto.
- **Permisos:** una cuenta de usuario normal. Una GPU necesita `--device` más un ID `--grant`, para que una tarjeta compartida no se utilice por accidente. Antes de cargar un modelo, un gobernador de VRAM comprueba que la memoria que ya está en uso más la necesidad medida del modelo quepan dentro de `--vram-budget` (predeterminado 26.624 MiB). Nunca expulsa nada; `--wait` realiza sondeos hasta que la carga encaja.
- **Secretos y telemetría:** no lee ninguna credencial y no recopila ni envía ninguna telemetría.
- **Pasos protegidos:** en los proyectos bloqueados para una referencia de ejecución, se rechazan las incrustaciones de descripción y cualquier aceptación de descripciones basada en el modelo a menos que se pase `--allow-multimodal-pool`, para que las consultas de la referencia no se vean influenciadas por los modelos que mide.

Informe una vulnerabilidad según lo descrito en [SECURITY.md](SECURITY.md).

## Estado

**v0.3.0, un prototipo público.** El código se prueba sin conexión, utilizando pequeños sustitutos para las bibliotecas del modelo.
- **Calibración:** el estudio se ejecuta con sus propios datos, en una RTX 5090, y establece los umbrales de duplicados, que son provisionales y se basan en un conjunto por tipo. Las cifras de VRAM son cifras conservadoras para toda la tarjeta, incluido el contexto CUDA, medidas en una RTX 5090; otras tarjetas y corpus pueden diferir. Los umbrales del código se basan en 17 copias insertadas. El método y los datos se encuentran en [el número 16 del boletín](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Proyectos grandes:** los duplicados, los valores atípicos y el canal de búsqueda funcionan en bloques delimitados. En conjuntos sintéticos, la búsqueda de duplicados cercanos y el canal de búsqueda tardan minutos con 100.000 imágenes, mientras que antes tardaban horas o se quedaban sin memoria. La comprobación de duplicados cercanos de la puerta de descripción utiliza un filtro de prefijo exacto, por lo que solo compara las descripciones que podrían alcanzar 0,85. El primer escaneo es aproximadamente 2 veces más rápido por imagen, con resultados idénticos, y se ejecuta en paralelo bajo `--threads`.
- **Espere que la línea de comandos cambie antes de la versión 1.0.**

El registro de diseño, su base de investigación y su historial de revisión se encuentran en la biblioteca de investigación del estudio: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Desarrollo

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI mantiene la cobertura de líneas en un 90% o más.

## Licencia

[MIT](LICENSE). Los modelos que puede cargar no forman parte de este paquete, y cada uno tiene su propia licencia; lea la tarjeta del modelo antes de usarlo.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
