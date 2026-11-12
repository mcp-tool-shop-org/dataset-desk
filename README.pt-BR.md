<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

## Por que

Um estilo LoRA aprende com suas imagens de treinamento e suas legendas. Dois defeitos sutis prejudicam ambos:
- **Repetições.** A mesma imagem salva duas vezes, ou duas renderizações com apenas alguns pixels de diferença, ensinam a mesma imagem duas vezes.
- **Descrições que não são descrições.** Uma legenda copiada do prompt de geração ou uma regra colada em 54 imagens não diz nada ao modelo sobre a imagem. Pior ainda, se esse texto for usado posteriormente para testar a pesquisa, ele mede a recuperação do prompt, e não a compreensão da imagem.

dataset-desk encontra ambos em uma pasta de projeto, sem alterá-la. Ele gera um relatório e uma fila de revisão para que alguém possa agir. **Nada é removido automaticamente.**

## O que ele faz

| Comando | O que você obtém |
|---|---|
| `scan` | Examina o projeto em `scan.jsonl` e `scan-report.json` sob `--out` sem carregar um codificador |
| `index` | Examina o projeto (registros, metadados de onda, dados de geração PNG) e incorpora cada imagem exclusiva em um cache sob `--out` |
| `dupes` | Cópias byte a byte (sha256), quase duplicatas (um dHash de 256 bits com reconhecimento de espelhamento) e duplicatas semânticas (cosseno de incorporação), em formato JSON. Em um índice DINOv2, ele usa uma regra calibrada de dois sinais. Duas visualizações do mesmo objeto nunca são pareadas. |
| `search` | As imagens mais próximas de uma descrição (`--text`), uma imagem (`--image`) ou ambas juntas |
| `coverage` | As melhores imagens para cada assunto em uma lista de assuntos desejados, para que as lacunas fiquem visíveis |
| `report` | Um relatório em Markdown: duplicatas, as imagens aprovadas menos típicas por conteúdo e aparência, a auditoria da descrição e se cada descrição encontra sua própria imagem |
| `sidecar-import` | Importa pastas de imagens pareadas com arquivos de legenda `.txt` com o mesmo prefixo (`--sidecars` em `index`/`report`/`compose` faz o mesmo), executa o filtro de descrição, relata imagens sem um arquivo `.txt` sem inventar texto e grava tudo sob `--out` |
| `compose` | Compõe legendas finais de treinamento em `--out` a partir de regras por projeto e edições importadas (por padrão, apenas registra registros sinalizados na fila de revisão ou oferece rascunhos lado a lado com `--all`, sem substituir o texto humano ou do arquivo), escrevendo um manifesto de conjunto de dados de nível 3 (`{"items":[{"image","sha256","caption"}]}`) com a origem por parte |
| `import-edits` | Reimporta linhas editadas de `description-worksheet.md` para `--out/seat-edits.jsonl` com um ID de escritor `seat:<name>` |
| `text-index` | Incorpora os arquivos de texto e código de uma pasta `--source` (ou um checkout de rnd-data), contando os trechos com o tokenizador do próprio modelo |
| `text-dupes` | Arquivos exatos e quase duplicados. `--task code` compara apenas scripts. |
| `text-search` | Os arquivos mais próximos de uma consulta (`--text`). `--task code` usa o prompt de recuperação de código do modelo. |
| `lineage` | Versões do mesmo script em diferentes execuções, classificadas por similaridade |

**Os limites** vêm da calibração do estúdio em seus próprios dados. Cada um é marcado como provisório no código, com a medida de onde veio:
- **Imagens:** em um índice DINOv2, um par é uma duplicata quando `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Outros índices usam o dHash dentro de 26 bits, mais cosseno ≥ 0,95.
- **Texto:** 0,98 com bge-m3 e 0,95 para código com EmbeddingGemma 2. Qualquer outro modelo precisa de `--semantic`.

Duas renderizações do mesmo objeto de diferentes ângulos de câmera nunca são duplicatas. A visualização vem de cada registro `provenance.asset_id` e `render_id`, ou de nomes de arquivos como `<asset>__y<yaw>_e<elev>`.

**O filtro de descrição** é executado sem nenhum modelo. Uma descrição é sinalizada quando:
- ecoa seu prompt: é igual ao prompt ou está dentro dele, compartilha uma sequência de 6 ou mais palavras com ele, ou 90% ou mais de suas palavras (80% ou mais de suas palavras de conteúdo) aparecem nele;
- envolve a linha de assunto do registro: uma sequência contígua de 4 ou mais palavras `canon.subject` aparece;
- é muito curta: menos de 8 palavras ou menos de 5 palavras de conteúdo;
- duplica outra descrição no conjunto: idêntica após a remoção da pontuação, ou um Jaccard de palavras de conteúdo de 0,85 ou mais;
- é muito longa para SigLIP2: mais de 64 tokens (o único teste que precisa de um tokenizador; os registros não verificados nunca são marcados como aprovados);
- viola as regras `caption-rules.json` por projeto (quando presentes): `prefix_missing`, `required_term_missing` ou `banned_word` (junto com os testes de conjunto `exact_duplicate` e `near_duplicate`).

Os registros sinalizados vão para `description-review.jsonl` e para uma **planilha de trabalho anônima**: cada imagem sinalizada ao lado de 3 a 5 irmãos visualmente mais próximos (escolhidos da similaridade apenas de imagem DINOv2 quando armazenados em cache sob `--out`, caso contrário dHash; nunca incorporações de texto em um conjunto bloqueado), sem prompt, sem linha de assunto e sem texto antigo visível, para que a pessoa que o reescreva descreva a imagem. Um redator opcional (`--draft plain-sight` por padrão ou `--draft qwen3-vl`) pode escrever um primeiro rascunho a partir dos pixels em si na fila de revisão para o conjunto (nunca diretamente na legenda de treinamento); a saída bruta completa do redator é salva por imagem em `drafter-raw.jsonl`. A aprovação em `--all` também cria rascunhos de registros não sinalizados como ofertas lado a lado (`draft_offer` / `draft_offer_writer`) sem substituir o texto humano ou do arquivo. Os rascunhos editados pelo conjunto são registrados como `captioner_assisted`.

**O canal de aparência** é composto por 30 números por imagem (paleta, tom e textura na resolução nativa), para que "a renderização desta está errada" seja mostrada separadamente de "o assunto desta é estranho".

## Instalação

```bash
pip install dataset-desk
```

Isso fornece a **versão lite**: apenas numpy e Pillow, sem torch. As verificações de duplicatas, o canal de aparência, o filtro de descrição e o relatório funcionam nela (`--backend none`).

Para o índice de incorporação (pesquisa, cobertura, duplicatas semânticas, a verificação de ajuste da descrição), adicione as bibliotecas do modelo:

```bash
pip install "dataset-desk[models]"
```

No Windows com uma GPU NVIDIA, instale primeiro uma versão CUDA do torch de `https://download.pytorch.org`; o torch do Windows de PyPI é apenas para CPU.

Os modelos são obtidos do Hugging Face em revisões fixas na primeira vez que são usados e, em seguida, carregados do seu cache. Cada tarefa tem um padrão e `--backend` escolhe outro:

| Backend | Modelo | Padrão para |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | texto |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | código (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

Redatores de imagens anônimos (`--draft`, `dataset_desk.registry.DRAFTER_REGISTRY`; ambos declaram `is_arm_model=False`, portanto, os conjuntos de teste bloqueados permitem que eles criem rascunhos):

| Redator | Modelo | Fixar | Licença | Padrão |
|---|---|---|---|---|
| `plain-sight` | [Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317` | MIT | sim |
| `qwen3-vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | — |

EmbeddingGemma 2 é usado da forma que o seu ficheiro de modelo indica: um prefixo de tarefa por comando, carregamento apenas de texto para texto e `--dim 768|512|256`. Defina `HF_HUB_OFFLINE=1` para proibir downloads. `plain-sight` está sempre offline; `qwen3-vl` respeita `HF_HUB_OFFLINE` (`local_files_only=True`), recusa qualquer incompatibilidade de revisão (`MODEL_REVISION_MISMATCH`), respeita o controlador de VRAM, aceita um prompt por projeto (`drafter_prompt` ou `--drafter-prompt`) e armazena o prompt e a saída bruta completa por imagem.

**Executa em:** Python 3.11 e 3.12, no Windows e Linux. CPU por padrão, com cada pool de threads limitado (4 threads, a menos que `--threads` indique o contrário), e a execução de um modelo reduz o processo para uma prioridade abaixo do normal, para que possa partilhar uma máquina.

## Utilização

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

### Regras de legenda por projeto (`caption-rules.json`) e proveniência

Coloque `caption-rules.json` em `<project>/caption-rules.json` (ou passe `--caption-config path/to/rules.json`):

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

- **Verificações de qualidade (sem modelo, relatadas em `description-review.jsonl`):** `prefix_missing` (a legenda não começa com `prefix`), `required_term_missing` (qualquer entrada `required_terms` para a classe do registro — obtida de `class_field` ou `class_pattern` — ou um `required` de fundo está ausente), `banned_word` (qualquer frase global, de fundo ou de classe `banned_words` aparece), mais o conjunto `exact_duplicate` e `near_duplicate` (Jaccard ≥ 0,85).
- **Composição (`compose` ou `report --compose`):** escreve `<out>/captions-manifest.json` (`--manifest-name` substitui o nome do arquivo) no formato `style-dataset-lab` lane-3 `run_dataset` `{"items":[{"image","sha256","caption",...}]}` — nunca dentro de `--project`. Cada legenda é montada a partir de `prefix`, termos obrigatórios `class`, termos opcionais `backdrop` e a parte de conteúdo visível livre (`seat` editado quando importado, caso contrário, o texto aprovado humano/sidecar — nunca um rascunho não editado). Por padrão, `compose` e `report --compose` rascunho **apenas** registra os itens sinalizados pela verificação de qualidade sem modelo na fila de revisão; `--all` também escreve ofertas opcionais `draft_offer` / `draft_offer_writer` ao lado dos originais não sinalizados, sem substituir o texto humano ou sidecar.
- **Registro de proveniência:** cada legenda final em `captions-manifest.json` e `description-review.jsonl` contém uma lista `writers` por parte, usando `config:<file>:<line>`, `drafter:<id>@<rev>:<prompt sha>`, `seat:<name>`, `record:<file>` e `sidecar:<file>`, com o nível `captioner_assisted` quando editado por um usuário após visualizar um rascunho ou oferta, `"human"` para registros humanos e `"sidecar"` para `.txt` sidecars.
- **Importação de sidecar (`sidecar-import` ou `--sidecars`):** associa cada imagem ao seu sidecar com o mesmo nome base `.txt`, atribui `author_kind: "sidecar"` (nunca `"human"`), extrai os prompts de geração PNG incorporados ComfyUI quando presentes, executa a verificação de qualidade completa (eco do prompt, duplicatas exatas/Jaccard e `caption-rules.json`), relata qualquer imagem sem um sidecar `.txt` em vez de inventar texto e escreve todos os registros importados e arquivos de revisão estritamente em `--out`.
- **Bloqueio do conjunto por tag, proteção de cópia renomeada com falha fechada e status `pool_detection`:** `check_multimodal_pool` aplica o bloqueio de referência em ambos os comandos `--project` e de texto (`text-index`, `text-dupes`, `text-search`, `lineage`) quando o nome da pasta `realpath` ou `"multimodal_pool"` na configuração (`caption-rules.json` ou `project.json`) nomeia um conjunto bloqueado. Para pastas não marcadas fora desses nomes, ele calcula o hash do conjunto exato de imagens que a verificação lê (incluindo links simbólicos e junções de diretório do Windows) em relação a um manifesto local não vazio (`DATASET_DESK_POOL_MANIFEST` ou `pool_manifest` em `~/.config/dataset-desk/config.json`, mantido fora do repositório) ou `--locked-pool-dir <path>` — e, nos comandos de texto, também verifica os nomes base do sidecar da legenda `.txt` em relação aos nomes base das imagens do conjunto bloqueado — recusando com falha fechada (`PERM_LOCKED_POOL`), nomeando o conjunto correspondente e a contagem. Cada comando resolve um status `pool_detection` (`status: "on"` com >= 1 hash de imagem por conjunto configurado e contagens por fonte — nunca valores de hash — ou `status: "off"` com o texto de aviso, emitido uma vez para `stderr`) e registra `pool_detection` em `<out>/pool-detection.json`, saídas JSON/JSONL em `--out` e uma nota de uma linha em cada relatório Markdown.
- A detecção de cópia renomeada corresponde às imagens por `sha256`, portanto, nem uma cópia feita apenas de novas imagens (nenhuma compartilhada com um conjunto bloqueado) nem uma cópia redimensionada, recodificada ou com metadados removidos correspondem por `sha256`; marque esses projetos com `"multimodal_pool"` em `caption-rules.json` ou `project.json`.

`dataset-desk --help` lista todas as flags. Códigos de saída: `0` ok, `1` um erro do utilizador ou uma recusa, `2` um erro inesperado. Os erros são impressos como `error <CODE>: <message>` com uma linha `hint:`; `--debug` adiciona o rastreamento da pilha.

**O layout do projeto que ele lê** é o de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Um projeto sem `records/` também funciona, desde que as imagens estejam em pastas sob `outputs/` (os ficheiros no topo de `outputs/` são ignorados como folhas de contacto): cada imagem é "não registada" e as verificações de duplicados e de aparência ainda são executadas.

## Modelo de ameaças e confiança

- **O que ele lê:** a pasta do projeto que você nomeia com `--project`: imagens, `records/*.json`, metadados de onda e os dados de geração incorporados em ficheiros PNG. Os comandos de texto leem a pasta `--source` ou o checkout `--rnd-data` que você nomeia. Eles recusam o ficheiro privado e os armazenamentos de blobs do rnd-data e imprimem apenas caminhos relativos.
- **O que ele escreve:** apenas em `--out`: o seu cache de verificação, caches de incorporação, o relatório, `description-review.jsonl` e a folha de cálculo. Ele recusa um `--out` dentro do projeto (`PERM_OUT_INSIDE_PROJECT`), para que um projeto nunca seja modificado.
- **O que ele nunca toca:** nada fora dessas duas pastas. Ele não exclui nada, em nenhum lugar.
- **Rede:** a camada lite nunca a usa. Um backend de modelo obtém os seus pesos fixos do Hugging Face na primeira utilização, a menos que `HF_HUB_OFFLINE=1` seja definido; o gerador de rascunhos carrega com `HF_HUB_OFFLINE=1` e `local_files_only=True` sempre. Nenhum modelo executa código remoto.
- **Permissões:** uma conta de utilizador normal. Uma GPU precisa de `--device` mais um ID `--grant`, para que uma placa partilhada não seja utilizada por engano. Antes de carregar um modelo, um controlador de VRAM verifica se a memória já em uso mais a necessidade medida do modelo se encaixa em `--vram-budget` (padrão 26.624 MiB). Ele nunca remove nada; `--wait` verifica até que o carregamento se encaixe.
- **Segredos e telemetria:** ele não lê credenciais e não coleta nem envia telemetria.
- **Passos protegidos:** em projetos bloqueados para uma referência em execução, as incorporações de descrição e qualquer aceitação de descrições baseada em modelo são recusadas, a menos que `--allow-multimodal-pool` seja passado, para que as consultas da referência não sejam moldadas pelos modelos que ele mede.

Relate uma vulnerabilidade conforme descrito em [SECURITY.md](SECURITY.md).

## Estado

**v0.3.0, um protótipo público.** O código é testado offline com pequenas amostras para as bibliotecas do modelo.
- **Calibração:** o programa é executado no estúdio com seus próprios dados, em uma RTX 5090, definindo os limites de duplicação, que são provisórios e baseiam-se em um conjunto por tipo. Os valores de VRAM são valores conservadores para toda a placa, incluindo o contexto CUDA, medidos em uma RTX 5090; outras placas e conjuntos de dados podem apresentar resultados diferentes. Os limites do código baseiam-se em 17 cópias inseridas. O método e os dados estão disponíveis em [newsletter edição 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Projetos grandes:** as duplicações, os valores discrepantes e o canal de pesquisa funcionam em blocos delimitados. Em conjuntos sintéticos, a pesquisa de duplicatas e o canal de pesquisa levam minutos para 100.000 imagens, enquanto antes levavam horas ou excediam a memória disponível. A verificação de duplicatas do filtro de descrição usa um filtro de prefixo exato, portanto, compara apenas as descrições que podem atingir 0,85. A primeira análise é cerca de 2 vezes mais rápida por imagem, com resultados idênticos, e é executada em paralelo em `--threads`.
- **Espere que a linha de comando seja alterada antes da versão 1.0.**

O registro de design, sua base de pesquisa e seu histórico de revisão estão disponíveis na biblioteca de pesquisa do estúdio: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Desenvolvimento

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

O CI mantém a cobertura de código em 90% ou mais.

## Licença

[MIT](LICENSE). Os modelos que podem ser carregados não fazem parte deste pacote, e cada um tem sua própria licença; leia o cartão do modelo antes de usá-lo.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
