# The dataset desk: making and reviewing the studio's visual datasets (Pass 3)

**Status:** revised for Pass 3, Addendum, and Follow-up (2026-10-15).
- The prototype beside this file ([`desk.py`](desk.py), [`backends.py`](backends.py), [`description_gate.py`](description_gate.py), [`stopwords.txt`](stopwords.txt)) has 36 offline unit tests ([`test_desk.py`](test_desk.py)) using a fake embedder, a fake drafter, monkeypatched `plain-sight` offline drafter checks, and the torch-free `none` lite backend, plus a real-record gate check on `salt-road` and `rustline`.
- `python -W error -m unittest test_desk` passes (36 tests), and `ruff check --select F,E9,B desk.py backends.py description_gate.py test_desk.py` reports 0 findings.
- **Nothing has run on real art or on a real model yet.** Every model run below waits for the C0–C3 approvals under the rig's CPU and GPU rules (§5 and §7).

---

## 1. The check in three sentences

1. **The goal.** The studio's `style-dataset-lab` sets are the training data for its style LoRAs. One desk should help
   make and review them:
   - find byte-identical, perceptual (`dHash`), and semantic duplicates;
   - audit record descriptions with the **Revision 3 model-free acceptance gate** (`description_gate.py`, shared with
     `experiments/embeddinggemma-2/design-multimodal.md`) for prompt echoes, wrapped `canon.subject` strings, too-short
     texts, pool duplicates (`content-word Jaccard >= 0.85` or `"".join(words(a)) == "".join(words(b))`), and `> 64`
     SigLIP2 tokens before any counted retrieval benchmark or LoRA caption export runs;
   - search the art by description, by picture, or by interleaved picture + text;
   - show which wanted subjects are missing and pair them with canon exemplars for generation;
   - catch descriptions that don't match their images (display-only, accounting for co-described siblings);
   - flag images whose look (palette, tone, and native surface/brushwork texture) is off.
2. **The parts.** No single model does all of this, so each job goes to the tool or model that suits it:
   - **`description_gate.py` (stdlib only + optional lazy SigLIP2 tokenizer)** runs the Revision 3 description audit,
     exposes the frozen stopword list hash (`stopwords.txt`, SHA-256 `f04dffb1...ec2dd4e`), and builds the review queue
     (`description-review.jsonl`) and blind re-description worksheet (`description-worksheet.md`);
   - **`plain-sight` (`florence-community/Florence-2-large`, revision `4271c66b88cdbc05735372ec13b2360108de5317`)** drafts
     initial descriptions blind from the image file path alone (`--draft plain-sight`, `detail="medium"` /
     `<DETAILED_CAPTION>`, offline only via `plain_sight.engine.Florence2Engine` or native `transformers` classes),
     recorded with `author_kind: "captioner_assisted"` and the loaded `drafter_id` and `drafter_revision` for a person to
     rewrite into 15–35 word search prose;
   - **EmbeddingGemma 2** (or **SigLIP2**, or **`none`** in lite mode) is the index;
   - **a 30-number look channel** handles style: 24 HSV colour/tone statistics plus 6 native-resolution surface,
     brushwork and radial-FFT texture statistics ported from `style-dataset-lab/scripts/measure_image.py`, cached by
     `sha256` in `<out>/scan.jsonl` across backends, with a neural style descriptor (CSD / VGG19 Gram) once calibrated
     in C3;
   - **ai-eyes** (SigLIP2) verifies specific claims outside the guarded benchmark pool;
   - **`qwen_generate.py` / `comfy-headless`** executes waves and reference-steered edits, with `desk.py` reading their
     `generation.json`, `_curation.json` and ComfyUI PNG `tEXt` provenance receipts;
   - **a person** rewrites and approves.
3. **Why it helps the studio.** Everything runs locally and free. Each threshold is calibrated on the studio's own
   canon, not borrowed from web-scale papers.

---

## 2. Who does what

| part | job | why this tool / model | how it fails |
|---|---|---|---|
| `description_gate.py` + `stopwords.txt` (stdlib; lazy SigLIP2 tokenizer for 64-token cap) | Revision 3 acceptance tests, review queue (`description-review.jsonl`) & blind sibling worksheet (`description-worksheet.md`) | catches `salt-road`'s 54× repeated Director ruling and wrapped `canon.subject` strings, `rustline`'s 172→31 shared/echoed prompts, contiguous runs $\ge 6$, all-word share $\ge 0.90$, content-word share $\ge 0.80$, $< 8$ words or $< 5$ content words, content-word Jaccard $\ge 0.85$, and $> 64$ SigLIP2 tokens | lexical + token-length only; paraphrases with different words use the display-only `clustering` embedding check after the counted run |
| `plain-sight` (`florence-community/Florence-2-large`, rev `4271c66b`) | blind caption drafter (`--draft plain-sight`) receiving only the image file path | neither arm G (EG2) nor arm S (SigLIP2); uses native `transformers` classes offline (never hub-fetched remote code) and drafts pixel-only observations so the human editor never sees the generation prompt | Florence-2 drafts must be rewritten by a person into 15–35 word search prose (`author_kind: "captioner_assisted"`) |
| EmbeddingGemma 2 image + text (440M), or SigLIP2 so400m | the index: search, semantic duplicates, coverage, display-only description ↔ image agreement & `clustering` near-dupes, interleaved `<\|image\|>` + text queries | one space for text and images; EG2 has an 8,192-token multimodal window (`{70, 140, 280, 560, 1120}` soft vision tokens/image, default `280`) against SigLIP2's separate 64-token text tower | ranks mostly by **subject**, not look (§3); `float16` overflows activations (must run `float32` or `bfloat16`); MRL at `128d` drops 13.4 MMEB pts (floor locked at `≥256d`) |
| `dHash` (256 bits) | byte-identical and near-identical copies, plus selecting 3–5 nearest same-folder sibling images on the blind worksheet | general embedders are poor copy detectors (§3); `dHash` groups visually similar siblings without touching any benchmark arm model | misses heavy crops and recolours |
| palette, tone and native texture statistics (30 numbers in `[0, 1]`) | the baseline look channel (cached in `<out>/scan.jsonl`) | cheap, no model, sees colour/tone (`24` HSV numbers) plus native-resolution impasto/edge/FFT texture (`6` numbers from `measure_image.py`: `hf_ratio`, `flat_hf`, `edge_dens`, `edge_hard`, `flat_coh`, `band_hi`) | statistical summary only; doesn't understand mid-level painterly motifs |
| CSD ViT-L / VGG19 Gram (deferred to C3) | the learned style channel, if it beats the 30d baseline in C3 | trained/formulated for style similarity | CSD released weights may underperform the paper (§3) |
| ai-eyes (SigLIP2) | verifies specific caption claims outside the pre-run multimodal pool | discriminative; a different family from Florence-2 | **arm S** of the multimodal experiment; guarded by `desk.py` so it cannot accept or reject descriptions on `rustline`, `tallow-fen`, `salt-road`, or `hesperia` before the counted run |
| comfy-headless (v3.1.0) & `qwen_generate.py` | the make side: `qwen-image` waves with `LoraLoaderModelOnly`, `Qwen-Image-Edit-2511`, Union ControlNet | native to the studio's non-anime stack; writes `PIN_PER_STEP` receipts | `qwen_generate.py`'s `.convert("RGB").save()` strips PNG `tEXt` chunks, so `desk.py` reads both `generation.json` and PNG `tEXt` |
| a person | rewrites blind drafts into 12–45 word search prose (`judgment.explanation`); approves; owns the canon | canon-first | — |

---

## 3. Research grounding

Each line gives the source, what it found in our words, and the design implication.

**Duplicates and pruning**
- **SemDeDup** (Abbas et al. 2023, arXiv 2303.09540).
  - **Found:** on LAION, CLIP-space pairs above cosine 0.93–0.97 were treated as semantic duplicates. Removing up to
    about 37% left zero-shot accuracy unchanged, and which survivor was kept barely mattered.
  - **For the desk:** flag pairs from about 0.93 upward for review only, and recalibrate the cut for each embedder.
- **DataComp** (Gadre et al. 2023, arXiv 2304.14108).
  - **Found:** mild embedding dedup changed little, and aggressive dedup hurt. CLIP needed cosine 0.96 for 0.9
    precision on copies, at 0.02 recall; a copy-detection model did far better.
  - **For the desk:** keep a hash tier for copies, and use embeddings for same-scene duplicates only.
- **Somepalli et al. 2023** (arXiv 2212.03860) and **Carlini et al. 2023** (arXiv 2301.13188).
  - **Found:** duplicated training images get memorized, and models trained from scratch on small sets copy even
    without duplicates. Dedup reduced memorization but didn't remove it.
  - **For the desk:** dedup is necessary, not sufficient; after training, audit generations against the training set
    (deferred).
- **D4** (Tirumala et al. 2023, arXiv 2308.12284) and **Sorscher et al. 2022** (arXiv 2206.14486) `[secondary]`.
  - **Found:** with scarce data, keep the prototypical examples.
  - **For the desk:** dedup only, and never prune the canon's dense core.
- **The Vendi score** (Friedman and Dieng 2022, arXiv 2210.02410) `[secondary]`.
  - **For the desk:** report diversity as an effective number of distinct images. When $N > D$ ($D = 768$ or $1152$),
    compute eigenvalues on the dual covariance matrix $\frac{1}{N} X^T X$ ($D \times D$) rather than $\frac{1}{N} X X^T$
    ($N \times N$), which is mathematically identical and keeps eigendecomposition fast at 1,000s of images.

**Caption checking, bulk captioning, and style-LoRA trigger hygiene**
- **CLIPScore** (Hessel et al. 2021, arXiv 2104.08718) and **VQAScore** (Lin et al. 2024, arXiv 2404.01291).
  - **Found:** a cosine catches gross mismatches; VQAScore beats a cosine on compositional pairs (Winoground group
    46.0 against 7.8), but by much less on plain prompts.
  - **For the desk:** use the cosine as display-only triage, ranked rather than thresholded. Colour, count and relation
    errors need a local VLM later.
- **DALL-E 3's captions** (Betker et al. 2023) and **Brack et al. 2025** (arXiv 2506.16679).
  - **Found:** descriptive synthetic captions help, but a share of human text guards against the captioner's quirks.
    Dense captions trade away some diversity, and mixing caption lengths did best.
  - **For the desk:** keep the human review description, and add short and dense synthetic captions.
- **BLIP's CapFilt** (Li et al. 2022, arXiv 2201.12086) and **CHAIR** (Rohrbach et al. 2018, arXiv 1809.02156).
  - **For the desk:** a generator proposes and a discriminator disposes: `plain-sight` writes, `ai-eyes` checks. Treat
    Florence-2 captions as unaudited until then.
- **StyleDrop** (Sohn et al. 2023, arXiv 2306.00983) and kohya / `ai-toolkit` trigger decoupling.
  - **For the desk:** a style-LoRA training caption is `[trigger], [subject, pose, composition, lighting, local objects]`,
    with **all canon style/medium/palette/brushwork words stripped** so the LoRA weights absorb the visual style into
    the trigger token rather than leaking it into content words. Conversely, if a caption is a copy of the generation
    prompt (`prompt_echo`) or an administrative ruling (`salt-road`), the LoRA fails to bind distinct subjects.

**Style against content**
- **CSD** (Somepalli et al. 2024, arXiv 2404.01292), **MegaStyle** (Gao et al. 2026, arXiv 2604.08364), and
  **Frochte 2026** (arXiv 2605.09030).
  - **Found:** CLIP and SigLIP rank mostly by subject; style scores vary across artists.
  - **For the desk:** use set-relative ranks on two channels side by side (content and the 30d HSV + native-texture
    look channel), never fixed cutoffs.

**Software, hardware and model card specifics (2026-10-15 audit)**
- **EmbeddingGemma 2 model card & installed `transformers 5.19` / `sentence-transformers 6.1` source:**
  - **Task prefixes:** `query`, `document`, `symmetric` (`SentenceSimilarity`), `clustering`, and `classification`.
    All five are wired in `backends.EG2_PREFIX`. In `desk.py`, `query` embeds descriptions for image retrieval, while
    `clustering` embeds descriptions for display-only near-duplicate text grouping (`embedding_description_clusters`).
  - **Precision:** `float16` overflows activation dynamic range and yields `NaN` or silently degraded embeddings.
    `backends.py` blocks `float16`, runs `float32` on CPU, and uses `bfloat16` on CUDA (`--grant`) where supported,
    verifying finite unit-norm outputs after every encode.
  - **Matryoshka truncation:** MMEB v2 holds from `768d` (59.01) → `512d` (58.38) → `256d` (56.24), then drops 13.4
    points at `128d` (45.65). `backends.py` enforces `dim in (768, 512, 256)`.
  - **Vision soft tokens (`vision_tokens`):** In `transformers/models/gemma4/image_processing_gemma4.py`,
    `_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)` (default `280`). `EG2Embedder` validates against this tuple
    and wires `vision_tokens` into `SentenceTransformer` via `processor_kwargs={"image_seq_length": vision_tokens}` and
    `model[0].processing_kwargs = {"image": {"max_soft_tokens": vision_tokens}}` (plus setting
    `processor.image_processor.max_soft_tokens`).

---

## 4. What the prototype does (Pass 3)

```text
python desk.py index    --project P --backend eg2|siglip2|none [--limit N] [--threads 4] [--strict] [--images-only]
python desk.py dupes    --project P --backend eg2|siglip2|none
python desk.py search   --project P --backend eg2|siglip2 --text "a squat bog creature" [--image ref.png]
python desk.py coverage --project P --backend eg2|siglip2 --subjects wanted.txt
python desk.py report   --project P --backend eg2|siglip2|none [--subjects wanted.txt] [--images-only] [--draft none|florence2|fake]
```

- **It reads** a `style-dataset-lab` project, and never writes into one.
- **It writes** a shared scan cache (`<out>/scan.jsonl`), per-backend embedding caches (`<out>/<backend>/`), the
  description review queue (`<out>/description-review.jsonl`), the blind re-description worksheet
  (`<out>/description-worksheet.md`), and the Markdown report (`<out>/report-<backend>.md`) under `--out`.

### Key Pass 2 & Pass 3 mechanisms
1. **Pre-`numpy` thread-pool caps (`desk.py`, `backends.py`).**
   - `backends.cap_threads()` runs at the very top of `desk.py` and `backends.py` **before `import numpy`**, reading
     `--threads` from `sys.argv` (default `4`) so OpenBLAS/MKL/OMP pools cannot size themselves to all 24 cores during
     `img @ img.T` or FFTs.
   - Inside PyTorch regions (`backends.enter_torch_region()`), `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are set
     per the `embed-serve` disposition, and `torch.set_num_interop_threads(1)` is guarded by `_TORCH_INTEROP_SET` so it
     runs at most once per process.
2. **Peeking and arm-model acceptance guards (`check_multimodal_pool`, `ai_eyes_verify`, `eg2_fit_accept`).**
   - On `rustline`, `tallow-fen`, `salt-road`, and `hesperia`, `index` and `report` refuse to embed descriptions or run
     the embedding description check unless `--images-only` (or `--backend none`) or `--allow-multimodal-pool` is
     explicitly passed.
   - Before the multimodal experiment's counted run on those four projects, **no model that measures one of its arms
     may accept or reject a description**: EmbeddingGemma 2 is arm G (`eg2_fit_accept`) and `ai-eyes` is SigLIP2, which
     is arm S (`ai_eyes_verify`). Calling either without `--allow-multimodal-pool` raises `SystemExit` naming the
     project and the guarded arm model. The only acceptance tests on those four projects are the model-free gate and the
     person's edit.
3. **Record hygiene (`load_records`, `scan`).**
   - Loads `judgment.failure_modes` (list) and displays them beside `rejected` and `borderline` images in the report.
   - When `judgment.explanation` is non-empty on an `approved` record, that string is what the gate checks (so
     `salt-road`'s 54× repeated Director ruling in `judgment.explanation` is audited directly as a group of 54, while
     also checking whether `canon.subject` is wrapped). Rejection notes on non-approved records remain in
     `judgment_note` for display and are never embedded as search queries.
   - `captions()` and `swap_check()` run on `approved` images only.
   - When byte-identical files have different records, `_record_priority()` deterministically prefers `approved` over
     `borderline` over `rejected`, while logging the collision in `problems`.
4. **ComfyUI PNG `tEXt` graph tracer (`_png_prompt_meta`).**
   - Traces `KSampler` / `SamplerCustom` `positive` conditioning links back to their text node (`prompt`) and traces
     `negative` conditioning links to `negative_prompt`, rather than picking the longest `CLIPTextEncode` string.
5. **Shared `<out>/scan.jsonl` cache and look-channel timing.**
   - `sha256`, `dHash`, the 30d look vector, PNG metadata, and per-image compute time are cached in `<out>/scan.jsonl`
     and shared across all backends (`eg2`, `siglip2`, `none`, `fake`). `meta.json` records `look_seconds` and
     `look_seconds_per_image` so C0 states the full run's load before it starts.
6. **Revision 3 acceptance gate (`description_gate.py` + `stopwords.txt`).**
   - Implements the exact Revision 3 acceptance tests from `experiments/embeddinggemma-2/design-multimodal.md` with one
     rule set and no extra degrees of freedom (dropping Pass 2's 0.80 Jaccard cut and short-prompt exemption).
   - **Pinned arithmetic:**
     - Words are `re.findall(r"\w+", s.lower())`.
     - Content words are words not in [`stopwords.txt`](stopwords.txt) (212 frozen English stopwords;
       `STOPWORDS_SHA256 = "f04dffb10bdd8f7f712a31caabdee13a427f0f15fcd77b35cf6009ef0ec2dd4e"`).
     - A share is the fraction of the description's words (or content words), **counting repeats**, that appear in the
       prompt's word set (`word_share`).
     - Contiguous runs, and the `canon.subject` slice, are matched on these word sequences (`max_contiguous_run`,
       `contains_word_sequence`).
     - Content-word Jaccard is `|C(a) ∩ C(b)| / |C(a) ∪ C(b)|` on sets of content words (`content_jaccard`).
     - SigLIP2 token length is counted as
       `len(tokenizer(s.lower(), padding=False, truncation=False)["input_ids"])` on the cached
       `google/siglip2-so400m-patch14-384` tokenizer (revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`,
       `local_files_only=True`), special tokens included (`siglip2_token_count`, `siglip2_token_counts`). The processor's
       defaults pad and truncate to 64, so a count through them could never fail. The import is lazy so `stdlib`-only
       callers still import `description_gate.py`; any record whose token count wasn't computed is marked
       `gate_status = "unchecked"` and `passed = False`, never passed.
   - **A description fails if:**
     1. **It echoes its prompt (`prompt_echo`):**
        - it equals the record's generation prompt, or sits inside it (`words(desc)` as a contiguous slice of
          `words(prompt)`);
        - it shares a contiguous run of **6 or more words** with the prompt (`prompt_contiguous_run >= 6`);
        - **90% or more** of its words appear in the prompt (`prompt_word_share >= 0.90`);
        - **80% or more** of its content words appear in the prompt (`prompt_content_word_share >= 0.80`);
     2. **It wraps `canon.subject` (`wraps_canon_subject`):** the words of the record's `canon.subject`, when there are
        **4 or more**, appear as a contiguous slice of the description's words (`contains_word_sequence(words(desc), words(canon_subject))`);
     3. **It is too short (`too_short`):** fewer than **8 words**, or fewer than **5 content words**;
     4. **It duplicates another description in the pool:** for any pair across the whole pool,
        `"".join(words(a)) == "".join(words(b))` (`exact_duplicate`), or a Jaccard of **0.85 or more** between the two
        content-word sets (`near_duplicate`);
     5. **It is too long for SigLIP2 (`too_long_siglip2`):** more than **64 SigLIP2 tokens**.
7. **Blind sibling worksheet (`description-worksheet.md`) & `plain-sight` drafting (`--draft`).**
   - `description-worksheet.md` is **blind to the old text and the prompt**: it never shows the current description,
     `canon.subject`, or `provenance.prompt`.
   - For each flagged record, the worksheet shows:
     - the record `id` and the target image path;
     - **3–5 sibling images from the same folder**, chosen by nearest `dHash` Hamming distance (`select_siblings`) so
       the reviewer sees the local visual neighbourhood and writes a discriminating description;
     - the instruction: *"describe this picture in 15–35 words so a person could pick it out from the siblings shown"*;
     - `author_kind` (`captioner_assisted` when drafted, or `human`), the `Florence-2 draft` slot, and the `edited text`
       slot for the person's 12–45 word rewrite.
   - When `--draft florence2` (or `--draft fake` in offline tests) is passed, `draft_flagged_records()` runs the drafter
     on each flagged record **passing only the image file path** (never the record JSON, `canon.subject`, or
     `provenance.prompt`), and records `author_kind: "captioner_assisted"`, `drafter_id`, and `drafter_revision` in both
     `description-review.jsonl` and `description-worksheet.md`. Existing passing descriptions carry
     `author_kind: "human"`.

### Separation of benchmark search prose vs LoRA training captions
- **Search prose (`judgment.explanation`):** The multimodal benchmark's query is the human-edited search prose (12–45
  words) stored by `style-dataset-lab` as the record's `judgment.explanation`. That is the only text written into or
  imported from `description-review.jsonl` and `description-worksheet.md`.
- **LoRA training captions (separate product):** Trigger-prefixed captions with canon style vocabulary stripped
  (`[trigger], [content]`, built by `style-dataset-lab/src/lib/captions.js`) are a separate downstream product for LoRA
  training exports; `dataset-desk` never writes them into review or import files.

---

## 5. The calibrations (the first experiments)

Each needs its own approval, and Argon's review of its harness, before it runs.
- **C0. A smoke run.** 20 images on the CPU per backend (`eg2`, `siglip2`), on a project outside the multimodal pool. It
  confirms the image path works and measures seconds per image (including `look_seconds_per_image` in `meta.json`), so
  the full-index load can be stated before it runs. This is harness debugging on a free local model, and it publishes
  no results.
- **C1. The semantic-duplicate cut, per embedder.**
  - Make augmented copies (crop, recolour, resize) of $N$ approved images in memory, and embed them with the
    originals.
  - Plot the cosine of these known duplicates against each image's nearest non-duplicate neighbour.
  - Pick the cut with high precision, then look at about 30 pairs on each side of it.
- **C2. The caption check.**
  - Text-disjoint swapped descriptions (`swap_check()`, cache only).
  - Edited descriptions (a colour or count changed): what does the cosine miss?
  - Later, a local VLM scoring $P(\text{yes})$ on the flagged pairs plus a random 10%.
- **C3. The look channel (corrected label sets from the record audit).**
  - **True off-style positive labels:** `salt-road`'s 19 look/register rejections (`off_canon` flat vector/poster art,
    cold dark fantasy register, and `palette_off_target` green cast) and `hesperia`'s 16 look-off `off_canon` records
    (glossy/sleek superhero rendering, excluding the 7 anatomical-crop rejections).
  - **On-style negative control (content vs style separation test):** `tallow-fen`'s 54 `style_residue_prop`
    rejections, whose records explicitly state *"style on-target"* (`style_consistency = 0.82`) and which were rejected
    solely because the trigger phrase literalized a lantern prop. A true style channel should score these **near** the
    approved set on look, while content/phrase-grounding (`ai-eyes` / `comfy-headless` Florence-2 `detect`) catches the
    leaked lantern.
  - **The runners:** rank by EG2, SigLIP2, the 30d palette+texture look channel, VGG19 Gram / DISTS, and CSD if
    downloaded.
  - **The score:** an AUC separating true off-style rejections from approved canon (and near-0.5 AUC on the on-style
    `tallow-fen` negative control).

**Peeking and arm-model guards.** The approved EG2 vs SigLIP2 multimodal experiment
(`experiments/embeddinggemma-2/design-multimodal.md`, Revision 3) runs one counted pass on all four projects
(`rustline`, `tallow-fen`, `salt-road`, and `hesperia`) after the desk stage. Until that counted run is done:
1. `desk.py` refuses to embed descriptions or run `captions()` on those four projects unless `--images-only` (or
   `--backend none`) or `--allow-multimodal-pool` is used;
2. Neither arm G (`EmbeddingGemma 2`, `eg2_fit_accept`) nor arm S (`SigLIP2` / `ai-eyes`, `ai_eyes_verify`) may accept
   or reject a description on those four projects. Only the model-free Revision 3 gate (`description_gate.py`) and the
   human editor's rewrite decide acceptance.

---

## 6. Director questions & Revision 3 contracts

### 6A. Where the desk lives: `mcp-tool-shop-org/dataset-desk` vs `rnd`

**Answer: Yes, the package boundary is right.** Moving the tool to `mcp-tool-shop-org/dataset-desk` as an installable
Python package with a CLI while keeping the design record, Argon reviews, and calibration/benchmark harnesses in `rnd`
cleanly separates **reusable studio tooling** from **experimental protocols and counted runs**.

**Recommended module layout in `mcp-tool-shop-org/dataset-desk`:**

| module | dependencies | responsibility |
|---|---|---|
| `dataset_desk/description_gate.py` + `stopwords.txt` | Python stdlib only (lazy `transformers` tokenizer for 64-token check) | Revision 3 acceptance tests, `STOPWORDS_SHA256`, blind sibling worksheet writer, and pixel-only draft attachment (`draft_flagged_records`). Zero required third-party deps so `rnd` harnesses and `style-dataset-lab` can import it anywhere |
| `dataset_desk/scan.py` | `numpy`, `Pillow` | Project scanner: `records/*.json`, `outputs/<wave>/generation.json`, `_curation.json`, ComfyUI PNG `tEXt` sampler-link tracer, `sha256`, 256-bit `dHash`, 30d HSV + native surface/brushwork/FFT look channel, and shared `<out>/scan.jsonl` cache |
| `dataset_desk/backends.py` | `numpy` (`torch` / `onnxruntime` lazy per tier) | Pre-`numpy` thread-pool caps (`cap_threads`, `enter_torch_region`), `NoneEmbedder` (`lite`), `FakeEmbedder` & `FakeDrafter` (tests), `PlainSightDrafter` (`Florence-2-large`), optional `OnnxSigLIP2LiteEmbedder` (`lite+onnx`), and lazy-loaded `EG2Embedder` / `SigLIP2Embedder` (`full`) |
| `dataset_desk/analyses.py` | `numpy` | Cache-only analyses: 3-tier `dupes`, `vendi` (with dual covariance for $N > D$), `outliers`, `look_fit`, `reading`, `captions` (with `group_rank`), text-disjoint `swap_check`, `embedding_description_clusters`, `search`, `coverage` |
| `dataset_desk/report.py` & `cli.py` | stdlib + above | Markdown report generator (`report-<backend>.md`), arm-model guards, and `dataset-desk` CLI entry point |

**What stays in `rnd`:**
- `experiments/dataset-desk/design.md`, review briefs, and C0–C3 calibration scripts and data tables;
- `experiments/embeddinggemma-2/design-multimodal.md` and its counted retrieval harness (which imports
  `dataset_desk.description_gate` directly so the desk's gate and the experiment's query gate are the exact same code).

---

### 6B. `dataset-desk-lite` for machines without recent CUDA or heavy RAM

#### 1. Are tiers in one codebase right, or should `lite` be its own package?
**One codebase with optional install tiers (`dataset-desk`, `dataset-desk[onnx]`, `dataset-desk[full]`) is right — not a
separate package.**
- `description_gate.py` (stdlib) and `scan.py` (`numpy` + `Pillow`) already power 100% of the model-free desk (`--backend none`): exact and `dHash` duplicates, the 30d HSV + native surface/brushwork/FFT look channel, record/wave/PNG provenance, Problems, and the model-free description audit + blind sibling worksheet.
- A forked `dataset-desk-lite` repository would immediately duplicate record parsing, the 30d look channel, and the report format, creating version drift against `style-dataset-lab`.
- In the prototype today, `--backend none` implements and tests the torch-free `lite` tier with zero new dependencies.

#### 2. Which small embedder and runtime fit `lite` (with what licence and RAM)?
We evaluated the three candidate options:

| candidate | runtime | disk / peak RAM | evaluation for `lite` |
|---|---|---|---|
| **1. `google/siglip2-base-patch16-256` (INT8 ONNX)** *(Recommended)* | `onnxruntime` CPU (MIT) | ~100 MB disk (vision + text INT8 ONNX); **~280–320 MB peak RAM** | **Best fit.** 86M ViT-B/16 + 64-token text tower (`768d`), **Apache-2.0** licence. Same SigLIP2 family and sigmoid training objective as `ai-eyes` (`siglip2-so400m`), fixed patch count (256 tokens/image), ~45–80 ms/image on 4 AVX2 CPU threads, no PyTorch required |
| **2. EmbeddingGemma 2 at `256d` MRL** | PyTorch or ONNX | ~1.7 GB (FP32) / ~520 MB (INT8); **~850 MB–1.8 GB RAM** | **Does not solve the `lite` constraint.** MRL truncation (`768d -> 256d`) only slices the final linear projection; it still runs the full 440M Gemma-3n backbone and vision encoder on every image |
| **3. MobileCLIP-S1 / OpenCLIP ViT-B/32 (INT8 ONNX)** | `onnxruntime` CPU | ~95–150 MB disk; ~250 MB RAM | Fast, but 77-token CLIP text tower is weaker on dense studio descriptions and introduces a third embedding family unrelated to `eg2` or `siglip2` |

- **Recommendation for `lite` with an embedder (`--backend siglip2-lite`, deferred until after C0–C3):**
  Export `google/siglip2-base-patch16-256` (`768d`, **Apache-2.0**) to dynamic INT8 ONNX (`vision_model_quantized.onnx`
  + `text_model_quantized.onnx`) running on `onnxruntime` CPU (**MIT** licence). It needs no PyTorch, no CUDA, ~100 MB
  of weights, and ~300 MB of RAM.

#### 3. Can the cache and report stay compatible across tiers?
**Yes, completely:**
- **Shared project-level artifacts under `<out>/`:** `scan.jsonl` (`sha256`, `dHash`, 30d look vector, PNG metadata,
  `look_seconds`), `description-review.jsonl`, and `description-worksheet.md` live at the root of `--out` and are
  shared across `--backend none`, `siglip2-lite`, `siglip2`, and `eg2`. A machine running `--backend none` builds
  `scan.jsonl` and the description review queue; a workstation running `eg2` later reuses `scan.jsonl` without
  rescanning a single image.
- **Per-backend vector caches under `<out>/<backend>/`:** `images.npy`, `texts.npy`, `cluster_texts.npy`, and `meta.json`
  sit in `<out>/none/`, `<out>/siglip2-lite/`, `<out>/siglip2/`, and `<out>/eg2/`. Because cosine thresholds are
  backend-specific, isolating `.npy` matrices by backend name prevents silent cross-model mixing while keeping an
  identical schema.
- **Compatible reports:** `report-<backend>.md` uses the exact same headings and tables across all tiers; when
  `--backend none` or `--images-only` is used, embedding-dependent sections remain present with an explicit
  `skipped (...)` note.

---

### 6C. Importer contract (`style-dataset-lab`) and `author_kind` provenance

1. **One shared Revision 3 gate, no extra degrees of freedom.**
   - All acceptance thresholds are fixed in `description_gate.py` per Multimodal Design Revision 3 (§4.6 above):
     prompt-echo (`inside_prompt`, `prompt_contiguous_run >= 6`, `prompt_word_share >= 0.90`,
     `prompt_content_word_share >= 0.80`), `wraps_canon_subject` (the words of `canon.subject`, when $\ge 4$, appear as
     a contiguous slice of `words(desc)`), `too_short` ($< 8$ words or $< 5$ content words), pool duplicate
     (`"".join(words(a)) == "".join(words(b))` or `content_jaccard >= 0.85`), and `siglip2_tokens > 64` counted via
     `len(tokenizer(s.lower(), padding=False, truncation=False)["input_ids"])` on the cached
     `siglip2-so400m-patch14-384` tokenizer (with uncounted records marked `"unchecked"` and unable to pass).
2. **`style-dataset-lab` importer contract and `author_kind`.**
   - `dataset-desk` remains strictly read-only on the project and writes `<out>/description-review.jsonl` and
     `<out>/description-worksheet.md`.
   - Every description carries `author_kind`:
     - `human`: existing passing descriptions written by a person (such as the passing `tallow-fen` and `hesperia`
       texts);
     - `captioner_assisted`: every re-description drafted by `plain-sight` (`Florence-2`) and rewritten by a person into
       12–45 words of search prose (`drafter_id` and `drafter_revision` are recorded alongside `author_kind`).
   - When `style-dataset-lab` imports completed edits (`sdlab descriptions import --project <P> --from <out>/description-worksheet.md`), it:
     1. Verifies that every imported re-description has been rewritten by a person (12–45 words, not clicked through
        unchanged from the raw `Florence-2 draft`) and runs `description_gate.audit_records()` with the SigLIP2
        tokenizer active to confirm `gate_status == "passed"` and `0` gate failures across the entire project pool;
     2. Writes the edited search prose to `records/<id>.json` as `judgment.explanation` (leaving LoRA training caption
        trigger formatting to `captions.js` at export time, never storing training captions in `judgment.explanation`);
     3. Writes `author_kind` (`"human"` or `"captioner_assisted"`), `drafter_id`, `drafter_revision`, and an immutable
        entry in `provenance.description_history` (`timestamp`, `author`, `author_kind`, `previous_explanation`,
        `stopwords_sha256`, `gate_receipt`).

---

## 7. Resource rules

- **CPU by default:** 4 threads, every pool capped before `numpy` and `torch` load (`cap_threads()` at top of `desk.py`
  and `backends.py`, plus `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` inside `enter_torch_region()`), below-normal
  process priority. Heavy index runs are launched under `start /AFFINITY 3FC /BELOWNORMAL`, on efficiency cores. Their
  load and duration are stated before they start, from C0's numbers (`look_seconds_per_image` and `image_seconds` in
  `meta.json`).
- **A GPU only under a Publisher grant** (`--grant`, recorded in `meta.json`; uses `bfloat16` on CUDA 13.4.3 / RTX 5090).
- **The environment:** `E:\AI\envs\embedgemma` (torch 2.14.1+cpu, transformers 5.19.0, sentence-transformers 6.1.0).
  Both models are already cached, at pinned revisions.

---

## 8. Deferred, and what would bring each in

 keeping v0 focused on the review desk (`index`, `dupes`, `search`, `coverage`, the 30d look reading, the model-free
 `description_gate`, the display-only description check, and the report), the roadmap items below wait for their triggers:

| item | trigger |
|---|---|
| a CSD ViT-L backend (weights CC-BY-4.0; a download needs the Director's go) | C3's design is approved |
| `--backend siglip2-lite` (`google/siglip2-base-patch16-256` INT8 ONNX on `onnxruntime` CPU) | packaging into `mcp-tool-shop-org/dataset-desk` after C0–C3 |
| a local VLM for VQAScore-style compositional checks | C2 shows what the cosine misses |
| `ai-eyes` noun verification outside the guarded benchmark pool + `captions.js` trigger stripping for LoRA export | after the multimodal experiment's counted run on the four benchmark projects |
| SSCD or another learned copy detector | `dHash` misses copies in C1 |
| the Vendi score per subject (needs subject labels) | a subject list exists for a project |
| "image minus caption" content-reduced vectors (InstantStyle) | C3 shows the look channels are weak |
| **Five-stage closed loop — Stage 1: two-LoRA tier orchestration** (exploratory wave LoRA vs canon lock LoRA) | after C3 validates the look channel on real waves |
| **Five-stage closed loop — Stage 2: medoid triage** (k-medoids / facility location over waves before human review) | wave sizes exceed ~200 candidates per run |
| **Five-stage closed loop — Stage 3: confidence-gated auto-placement** (routing candidates into review buckets) | C1 + C3 thresholds calibrated and approved by the Director |
| **Five-stage closed loop — Stage 4: gap-driven generation through `comfy-headless`** (dispatching `coverage()` gaps + exemplars to `qwen_generate.py` / `Qwen-Image-Edit-2511`) | after the review desk is in daily use on at least two projects |
| **Five-stage closed loop — Stage 5: grounding-based salvage** (Florence-2 `detect`/`segment` + localized inpaint for `style_residue_prop` plates like `tallow-fen`'s 54 on-style lantern rejections) | C3 confirms `tallow-fen`'s 54 `style_residue_prop` plates pass the look channel |
| a post-training memorization audit (top-1 training-set match for ~200 LoRA generations) | a LoRA is trained on a desk-curated set |
| an HTML contact sheet | the Markdown report is hard to review |

---

## 9. Standards compliance

| standard | score | evidence |
|---|---|---|
| PIN_PER_STEP | 2 | model revisions (`eg2` `914f7f8`, `siglip2` `e8e4872`, `florence2` `4271c66b`), `stopwords.txt` SHA-256 (`f04dffb1...ec2dd4e`), dimensions, dtype and discrete `vision_tokens` (`70, 140, 280, 560, 1120`) are pinned and recorded in `meta.json` / `description-review.jsonl`; `dHash`, the 30d look channel (`scan.jsonl`), and `description_gate.py` are deterministic |
| ANDON_AUTHORITY | 2 | stops on a missing cache, a write into the project, `float16` or `<256d` MRL or unsupported `vision_tokens` on EG2, `NaN`/`Inf` vectors, a multimodal-pool project without `--images-only` / `--allow-multimodal-pool`, an arm-model acceptance call (`ai_eyes_verify`, `eg2_fit_accept`) on the guarded pool, an uncounted SigLIP2 token length (`gate_status="unchecked"`), or a GPU without a grant |
| NAMED_COMPENSATORS | 2 | no irreversible calls. It never moves, deletes or edits project files; everything it writes is under `--out`, and deleting that folder undoes it |
| DECOMPOSE_BY_SECRETS | 2 | `description_gate` (stdlib), `backends`, `scan`, cache and analyses are separate; `draft_flagged_records()` passes only image file paths to the drafter, and `description-worksheet.md` withholds current text, `canon.subject`, and `provenance.prompt` |
| UNCERTAINTY_GATED_HUMANS | 2 | every output is a ranked prompt to look or a blind sibling worksheet requiring a human rewrite, never an automatic removal or silent overwrite |
| EXTERNAL_VERIFIER | n/a | no specialized claims |

---

## 10. Argon Pass 1, Pass 2 & Pass 3 review dispositions

1. **Pre-`numpy` thread caps & torch BLAS isolation (Pass 2 §2.1):** `backends.cap_threads()` runs before `import numpy`
   in both `desk.py` and `backends.py`; `enter_torch_region()` sets `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` and
   guards `torch.set_num_interop_threads(1)` once per process. Tested in `ThreadCaps`.
2. **Peeking & outcome-selection guard (Pass 2 §2.2):** Added `--images-only` and `--allow-multimodal-pool`; `desk.py`
   refuses to embed descriptions or run `captions()` on `rustline`, `tallow-fen`, `salt-road`, and `hesperia` without
   the flag. Tested in `test_peeking_and_arm_model_guards_refuse_salt_road_without_flag`.
3. **`failure_modes` list & rejection notes excluded from descriptions (Pass 2 §2.3–2.4):** `load_records()` loads
   `judgment.failure_modes` (list) and displays them beside `rejected` and `borderline` images; `explanation` is
   `canon.subject` or `judgment.explanation` **only** when `status == "approved"`; `captions()` and `swap_check()` use
   approved images only. Tested in `test_failure_modes_loaded_and_rejection_notes_not_used_as_descriptions`.
4. **`vision_tokens` wired into EG2 (Pass 2 §2.5):** Read `transformers 5.19` (`Gemma4ImageProcessor._SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)`) and `sentence-transformers 6.1` (`Transformer.processor_kwargs` and `Transformer.processing_kwargs`); wired both in `EG2Embedder`. Tested in `test_eg2_guards_vision_tokens_and_multimodal_embedder`.
5. **ComfyUI PNG prompt parser follows sampler links (Pass 2 §2.6):** `_png_prompt_meta()` traces the sampler's
   `positive` input link to `prompt` and `negative` input link to `negative_prompt`. Tested in
   `test_wave_generation_curation_and_png_sampler_positive_vs_negative`.
6. **Shared `<out>/scan.jsonl` cache & look timing (Pass 2 §2.7):** Keyed by `sha256` and shared across backends;
   `look_seconds` and `look_seconds_per_image` recorded in `meta.json`. Tested in
   `test_a_rerun_embeds_only_what_is_new_and_shares_scan_cache`.
7. **Deterministic record priority on byte-identical collisions (Pass 2 §2.8):** `_record_priority()` prefers
   `approved` over `borderline` over `rejected` regardless of directory walk order, and logs the collision in `problems`.
   Tested in `test_byte_identical_files_with_different_records_prefer_approved_deterministically`.
8. **Revision 3 acceptance tests & pinned arithmetic (`description_gate.py`, `stopwords.txt`, Pass 3 §1 & Pass 3 Addendum):**
   Replaced Pass 2's gate rules and short-prompt exemption with Revision 3's exact rules: frozen `stopwords.txt` with
   exposed SHA-256 (`STOPWORDS_SHA256`), `prompt_echo` (equals/inside prompt, contiguous run $\ge 6$, all-word share
   $\ge 0.90$, content-word share $\ge 0.80$), `wraps_canon_subject` (the words of `canon.subject`, when $\ge 4$, appear
   as a contiguous slice of `words(desc)`), `too_short` ($< 8$ words or $< 5$ content words), pool duplicate
   (`"".join(words(a)) == "".join(words(b))` or content-word Jaccard $\ge 0.85$), and lazy SigLIP2 token length
   (`len(tokenizer(s.lower(), padding=False, truncation=False)["input_ids"])` on the cached
   `siglip2-so400m-patch14-384` tokenizer, failing above 64 tokens; uncounted records marked `"unchecked"` and cannot
   pass). When `judgment.explanation` is non-empty, the gate audits `judgment.explanation` directly while also checking
   `canon.subject` wrapping. Tested on every exact boundary in `DescriptionGateTests`.
9. **Blind sibling worksheet (`description-worksheet.md`, Pass 3 §2):** Stripped current description, `canon.subject`,
   and `provenance.prompt` from the worksheet; added 3–5 nearest-`dHash` sibling images from the same folder
   (`select_siblings`), the instruction *"describe this picture in 15–35 words so a person could pick it out from the
   siblings shown"*, `Florence-2 draft`, `edited text`, and `author_kind`. Tested in
   `test_54_copy_ruling_inside_prompt_blind_worksheet_and_fake_drafter`.
10. **Blind `plain-sight` drafting (`--draft`, `PlainSightDrafter`, `FakeDrafter`, Pass 3 §3 & Pass 3 Follow-up):** Added
    `draft_flagged_records()` passing only image file paths to the drafter and recording
    `author_kind: "captioner_assisted"`, `drafter_id`, and `drafter_revision` (while existing passing descriptions carry
    `author_kind: "human"`). `PlainSightDrafter` pins `plain-sight`'s exact model (`florence-community/Florence-2-large`,
    revision `4271c66b88cdbc05735372ec13b2360108de5317`), lazily delegates to `plain_sight.engine.Florence2Engine`
    (with `detail="medium"` / `<DETAILED_CAPTION>`) or native `transformers` classes offline (`local_files_only=True`,
    `HF_HUB_OFFLINE=1`), never loads remote code, and raises a clear error naming the repo, revision, and
    `"install plain-sight"` if the snapshot is absent. Tested in
    `test_plain_sight_drafter_pin_offline_no_remote_code_and_missing_snapshot`.
11. **Arm-model acceptance guard (`ai_eyes_verify`, `eg2_fit_accept`, Pass 3 §4):** Guarded both arm S (`SigLIP2` /
    `ai-eyes`) and arm G (`EmbeddingGemma 2`) so neither can accept or reject descriptions on `rustline`, `tallow-fen`,
    `salt-road`, or `hesperia` without `--allow-multimodal-pool`. Tested in
    `test_peeking_and_arm_model_guards_refuse_salt_road_without_flag`.


