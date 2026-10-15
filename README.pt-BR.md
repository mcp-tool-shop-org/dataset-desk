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

O dataset-desk encontra ambos em uma pasta de projeto, sem alterá-la. Ele gera um relatório e uma fila de revisão para que uma pessoa possa agir. **Nada é removido automaticamente.**

## O que ele faz

| Comando | O que você obtém |
|---|---|
| `index` | Examina o projeto (registros, metadados de onda, dados de geração PNG) e incorpora cada imagem exclusiva em um cache em `--out` |
| `dupes` | Cópias byte a byte (sha256), quase duplicatas (256 bits dHash dentro de 10 bits) e duplicatas semânticas (cosseno de incorporação), em formato JSON |
| `search` | As imagens mais próximas de uma descrição (`--text`), uma imagem (`--image`) ou ambas juntas |
| `coverage` | As melhores imagens para cada assunto em uma lista de assuntos desejados, para que as lacunas fiquem visíveis |
| `report` | Um relatório em Markdown: duplicatas, as imagens aprovadas menos típicas por conteúdo e aparência, a auditoria da descrição e se cada descrição encontra sua própria imagem |

O **filtro de descrição** é executado sem nenhum modelo. Uma descrição é sinalizada quando:
- ecoa seu prompt: é igual ao prompt ou está dentro dele, compartilha uma sequência de 6 ou mais palavras com ele, ou 90% ou mais de suas palavras (80% ou mais de suas palavras de conteúdo) aparecem nele;
- envolve a linha de assunto do registro: uma sequência contígua de 4 ou mais palavras `canon.subject` aparece;
- é muito curta: menos de 8 palavras ou menos de 5 palavras de conteúdo;
- duplica outra descrição no conjunto: idêntica após a remoção da pontuação ou um índice de Jaccard de palavras de conteúdo de 0,85 ou mais;
- é muito longa para o SigLIP2: mais de 64 tokens (o único teste que requer um tokenizador; os registros não verificados nunca são marcados como aprovados).

Os registros sinalizados são enviados para `description-review.jsonl` e para uma **planilha de trabalho anônima**: cada imagem sinalizada ao lado de 3 a 5 imagens visualmente mais próximas, sem prompt, sem linha de assunto e sem texto antigo visível, para que a pessoa que a reescreva descreva a imagem. Um rascunhista opcional ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) pode escrever um primeiro rascunho apenas com os pixels; os rascunhos são registrados como `captioner_assisted`, e uma pessoa os edita.

O **canal de aparência** é composto por 30 números por imagem (paleta, tom e textura na resolução nativa), para que "a renderização desta imagem está errada" seja exibida separadamente de "o assunto desta imagem é estranho".

## Instalação

```bash
pip install "dataset-desk @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.0"
```

Isso fornece a **versão lite**: apenas numpy e Pillow, sem torch. As verificações de duplicatas, o canal de aparência, o filtro de descrição e o relatório funcionam nela (`--backend none`).

Para o índice de incorporação (pesquisa, cobertura, duplicatas semânticas, a verificação de ajuste da descrição), adicione as bibliotecas do modelo:

```bash
pip install "dataset-desk[models] @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.0"
```

Os modelos são obtidos do Hugging Face em revisões fixas na primeira vez que são usados e, em seguida, são carregados do seu cache: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`--backend eg2`) ou [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) (`--backend siglip2`). Defina `HF_HUB_OFFLINE=1` para proibir downloads. O rascunhista plain-sight é sempre offline: ele nunca faz download.

**Executa em:** Python 3.11 e 3.12, no Windows e Linux. CPU por padrão, com cada pool de threads limitado (4 threads, a menos que `--threads` diga o contrário), e uma execução de modelo reduz o processo para uma prioridade abaixo do normal, para que ele possa compartilhar uma máquina.

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

`dataset-desk --help` lista todas as sinalizações. Códigos de saída: `0` ok, `1` um erro do usuário ou uma recusa, `2` um erro inesperado. Os erros são impressos como `error <CODE>: <message>` com uma linha `hint:`; `--debug` adiciona o rastreamento da pilha.

O **layout do projeto que ele lê** é o de [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab):

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

Um projeto sem `records/` também funciona, desde que as imagens estejam em pastas sob `outputs/` (os arquivos no topo de `outputs/` são ignorados como folhas de contato): cada imagem é "não registrada", e as verificações de duplicatas e aparência ainda são executadas.

## Modelo de ameaças e confiança

- **O que ele lê:** a pasta do projeto que você nomeia com `--project`: imagens, `records/*.json`, metadados de onda e os dados de geração incorporados em arquivos PNG.
- **O que ele grava:** apenas em `--out`: seu cache de digitalização, caches de incorporação, o relatório, `description-review.jsonl` e a planilha de trabalho. Ele recusa um `--out` dentro do projeto (`PERM_OUT_INSIDE_PROJECT`), para que um projeto nunca seja modificado.
- **O que ele nunca toca:** qualquer coisa fora dessas duas pastas. Ele não exclui nada, em nenhum lugar.
- **Rede:** a versão lite nunca a usa. Um backend de modelo obtém seus pesos fixos do Hugging Face na primeira vez que são usados, a menos que `HF_HUB_OFFLINE=1` seja definido; o rascunhista é carregado com `HF_HUB_OFFLINE=1` e `local_files_only=True` toda vez. Nenhum modelo executa código remoto.
- **Permissões:** uma conta de usuário comum. Uma GPU precisa de `--device` mais um ID `--grant`, para que uma placa compartilhada não seja usada por engano.
- **Segredos e telemetria:** ele não lê credenciais e não coleta nem envia telemetria.
- **Etapas protegidas:** em projetos bloqueados para um benchmark em execução, as incorporações de descrição e qualquer aceitação de descrição baseada em modelo são recusadas, a menos que `--allow-multimodal-pool` seja passado, para que as consultas do benchmark não sejam moldadas pelos modelos que ele mede.

Relate uma vulnerabilidade conforme descrito em [SECURITY.md](SECURITY.md).

## Status

**v0.1.0, um protótipo público.** O código é testado offline (142 testes, 96% de cobertura de linha, com substituições pequenas para as bibliotecas de modelo), mas ainda não foi calibrado em dados reais: os limites de quase duplicatas e semânticas são pontos de partida. Espere que a linha de comando mude antes da versão 1.0.

O registo de design, a sua base de investigação e o histórico de revisões encontram-se na biblioteca de investigação do estúdio: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Desenvolvimento

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

O sistema de integração contínua (CI) garante uma cobertura de código de 90% ou mais.

## Licença

[MIT](LICENSE). Os modelos que pode carregar não fazem parte deste pacote, e cada um tem a sua própria licença; leia a ficha do modelo antes de o utilizar.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
