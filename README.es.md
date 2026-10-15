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

Un estilo LoRA aprende de sus imágenes de entrenamiento y sus descripciones. Dos defectos sutiles lo arruinan:
- **Repeticiones.** La misma imagen guardada dos veces, o dos renders separados por unos pocos píxeles, enseñan la misma imagen dos veces.
- **Descripciones que no son descripciones.** Una descripción copiada del mensaje de generación, o una regla pegada en 54 imágenes, no le dice nada al modelo sobre la imagen. Peor aún, si ese texto se utiliza más tarde para probar la búsqueda, mide la recuperación del mensaje, no la comprensión de la imagen.

dataset-desk detecta ambos, en una carpeta de proyecto, sin modificarla. Escribe un informe y crea una cola de revisión para que una persona actúe sobre ella. **Nada se elimina automáticamente.**

## Qué hace

| Comando | Qué se obtiene |
|---|---|
| `index` | Escanea el proyecto (registros, metadatos de onda, datos de generación PNG) e incluye cada imagen única en una caché en `--out` |
| `dupes` | Copias idénticas en bytes (sha256), duplicados cercanos (256 bits dHash dentro de 10 bits) y duplicados semánticos (coseno de incrustación), como JSON |
| `search` | Las imágenes más cercanas a una descripción (`--text`), una imagen (`--image`) o ambas juntas |
| `coverage` | Las mejores imágenes para cada sujeto en una lista de sujetos deseados, para que se muestren los espacios en blanco |
| `report` | Un informe en Markdown: duplicados, las imágenes aprobadas menos típicas por contenido y por apariencia, la auditoría de la descripción y si cada descripción encuentra su propia imagen |

**La puerta de enlace de la descripción** se ejecuta sin ningún modelo. Se marca una descripción cuando:
- hace eco de su mensaje: es igual al mensaje o está dentro de él, comparte una secuencia de 6 o más palabras con él, o el 90 % o más de sus palabras (el 80 % o más de sus palabras de contenido) aparecen en él;
- incluye la línea de asunto del registro: una secuencia continua de 4 o más palabras `canon.subject` aparece como una secuencia continua de sus palabras;
- es demasiado corta: menos de 8 palabras o menos de 5 palabras de contenido;
- duplica otra descripción en el conjunto: idéntica una vez que se elimina la puntuación, o un Jaccard de palabras de contenido de 0,85 o más;
- es demasiado larga para SigLIP2: más de 64 tokens (la única comprobación que necesita un tokenizador; los registros no comprobados nunca se marcan como válidos).

Los registros marcados se envían a `description-review.jsonl` y a una **hoja de trabajo ciega**: cada imagen marcada junto a 3 a 5 hermanos visualmente más cercanos, sin ningún mensaje, sin ninguna línea de asunto y sin ningún texto antiguo a la vista, para que la persona que la reescriba describa la imagen. Un redactor opcional ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) puede escribir un primer borrador solo a partir de los píxeles; los borradores se registran como `captioner_assisted` y una persona los edita.

**El canal de apariencia** es de 30 números por imagen (paleta, tono y textura en resolución nativa), por lo que "el renderizado de esta imagen es incorrecto" se muestra por separado de "el sujeto de esta imagen es extraño".

## Instalación

```bash
pip install "dataset-desk @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

Eso proporciona la **capa ligera**: solo numpy y Pillow, sin torch. Las comprobaciones de duplicados, el canal de apariencia, la puerta de enlace de la descripción y el informe funcionan en ella (`--backend none`).

Para el índice de incrustación (búsqueda, cobertura, duplicados semánticos, la comprobación de ajuste de la descripción), agregue las bibliotecas del modelo:

```bash
pip install "dataset-desk[models] @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

Los modelos se obtienen de Hugging Face en las revisiones fijadas la primera vez que se utilizan y, a continuación, se cargan desde su caché: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`--backend eg2`) o [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) (`--backend siglip2`). Establezca `HF_HUB_OFFLINE=1` para prohibir las descargas. El redactor de plain-sight siempre está desconectado: nunca descarga.

**Se ejecuta en:** Python 3.11 y 3.12, en Windows y Linux. CPU de forma predeterminada, con cada grupo de subprocesos limitado (4 subprocesos a menos que `--threads` indique lo contrario), y la ejecución de un modelo reduce la prioridad del proceso a por debajo de lo normal, para que pueda compartir una máquina.

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
```

`dataset-desk --help` enumera todas las marcas. Códigos de salida: `0` correcto, `1` un error de usuario o un rechazo, `2` un error inesperado. Los errores se imprimen como `error <CODE>: <message>` con una línea `hint:`; `--debug` agrega el rastreo de la pila.

**El diseño del proyecto que lee** es el de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Un proyecto sin `records/` también funciona, siempre y cuando las imágenes estén en carpetas dentro de `outputs/` (los archivos en la parte superior de `outputs/` se omiten como hojas de contacto): cada imagen es "no registrada" y las comprobaciones de duplicados y apariencia siguen funcionando.

## Modelo de amenazas y confianza

- **Lo que lee:** la carpeta del proyecto que nombra con `--project`: imágenes, `records/*.json`, metadatos de onda y los datos de generación incrustados en los archivos PNG.
- **Lo que escribe:** solo en `--out`: su caché de escaneo, cachés de incrustación, el informe, `description-review.jsonl` y la hoja de trabajo. Rechaza un `--out` dentro del proyecto (`PERM_OUT_INSIDE_PROJECT`), por lo que nunca se modifica un proyecto.
- **Lo que nunca toca:** nada fuera de esas dos carpetas. No elimina nada, en ninguna parte.
- **Red:** la capa ligera nunca la utiliza. Un backend de modelo obtiene sus pesos fijados de Hugging Face la primera vez que se utilizan, a menos que se establezca `HF_HUB_OFFLINE=1`; el redactor se carga con `HF_HUB_OFFLINE=1` y `local_files_only=True` cada vez. Ningún modelo ejecuta código remoto.
- **Permisos:** una cuenta de usuario normal. Una GPU necesita `--device` más un ID `--grant`, para que una tarjeta compartida no se utilice por accidente.
- **Secretos y telemetría:** no lee ninguna credencial y no recopila ni envía ninguna telemetría.
- **Pasos protegidos:** en los proyectos bloqueados para una prueba de rendimiento en curso, se rechazan las incrustaciones de descripción y cualquier aceptación de descripciones basada en el modelo a menos que se pase `--allow-multimodal-pool`, para que las consultas de la prueba de rendimiento no se vean influenciadas por los modelos que mide.

Informe una vulnerabilidad según lo descrito en [SECURITY.md](SECURITY.md).

## Estado

**v0.1.2, un prototipo público.** El código se prueba sin conexión (152 pruebas, 96 % de cobertura de líneas, con pequeños sustitutos para las bibliotecas del modelo), pero aún no se ha calibrado con datos reales: los umbrales de duplicados cercanos y semánticos son puntos de partida. Espere que la línea de comandos cambie antes de la versión 1.0.

El registro de diseño, la base de investigación y el historial de revisiones se encuentran en la biblioteca de investigación del estudio: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Desarrollo

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

La integración continua (CI) mantiene la cobertura de líneas en un 90 % o más.

## Licencia

[MIT](LICENSE). Los modelos que puede cargar no forman parte de este paquete, y cada uno tiene su propia licencia; lea la tarjeta del modelo antes de utilizarlo.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
