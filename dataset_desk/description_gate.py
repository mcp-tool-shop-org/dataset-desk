"""Model-free description gate, sibling selector, and blind review-worksheet builder for the dataset desk.

Implements the exact Revision 3 acceptance tests and pinned arithmetic shared with the EmbeddingGemma 2 multimodal
experiment harness (experiments/embeddinggemma-2/design-multimodal.md §2a & §6).

Core acceptance rules (stdlib only; no extra degrees of freedom):
A description fails if:
1. It echoes its prompt:
   - equals the record's generation prompt, or sits inside it;
   - shares a contiguous run of 6 or more words with the prompt;
   - 90% or more of its words appear in the prompt;
   - 80% or more of its content words appear in the prompt;
2. It wraps `canon.subject`: the words of the record's `canon.subject`, when there are 4 or more, appear as a
   contiguous slice of the description's words;
3. It is too short: fewer than 8 words, or fewer than 5 content words;
4. It duplicates another description in the pool: for any pair across the whole pool,
   `"".join(words(a)) == "".join(words(b))`, or a Jaccard of 0.85 or more between the two content-word sets;
5. It is too long for SigLIP2: more than 64 tokens, counted as
   `len(tokenizer(s.lower(), padding=False, truncation=False)["input_ids"])` on the cached
   `siglip2-so400m-patch14-384` tokenizer, special tokens included (computed via a separate function that imports the
   tokenizer lazily; a record whose token count was not computed is marked "unchecked" and cannot be marked as passing).

Pinned arithmetic:
- Words are `re.findall(r"\\w+", s.lower())`.
- A content word is a word not in the committed stopword list (`stopwords.txt`), whose sha256 is exposed by the gate.
- A share is the number of the description's words (counting repeats) that occur anywhere in the prompt, over the
  description's word count; the same for content words.
- Contiguous runs, and the `canon.subject` slice, are matched on these word sequences.
- Pool exact duplicates compare `"".join(words(a)) == "".join(words(b))`; near duplicates compare content-word set
  Jaccard `>= 0.85` across every pair in the pool.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import json.decoder
import math
import os
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from dataset_desk.errors import DeskError

STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")
CAPTION_RULES_FILENAME = "caption-rules.json"
CAPTION_MANIFEST_FILENAME = "captions-manifest.json"
SEAT_EDITS_FILENAME = "seat-edits.jsonl"
DRAFTER_RAW_FILENAME = "drafter-raw.jsonl"
SIDECAR_RECORDS_FILENAME = "sidecar-records.jsonl"
SIDECAR_REPORT_FILENAME = "sidecar-import.json"
SIDECAR_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".webp"})
KNOWN_MULTIMODAL_POOLS = frozenset({"rustline", "tallow-fen", "salt-road", "hesperia"})

# Pinned Revision 3 acceptance thresholds (no extra degrees of freedom)
MIN_WORDS = 8
MIN_CONTENT_WORDS = 5
MIN_CANON_SUBJECT_WORDS = 4
MAX_PROMPT_CONTIGUOUS_RUN = 6
MAX_PROMPT_WORD_SHARE = 0.90
MAX_PROMPT_CONTENT_WORD_SHARE = 0.80
MAX_CONTENT_JACCARD = 0.85
MAX_SIGLIP2_TOKENS = 64

WORKSHEET_INSTRUCTION = (
    "describe this picture in 15–35 words so a person could pick it out from the siblings shown"
)

_WORD_RE = re.compile(r"\w+")
_SAFE_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SEAT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_JSON_PRIM_RE = re.compile(r"(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_EPS = 1e-12


def prompt_sha256(prompt: str | None) -> str:
    """Return the 64-hex SHA-256 digest of a drafter prompt string (UTF-8)."""
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


def drafter_writer_id(drafter_id: str, drafter_rev: str, drafter_prompt: str | None = None) -> str:
    """Format the per-part provenance writer id `drafter:<id>@<rev>:<prompt sha>`."""
    return f"drafter:{drafter_id}@{drafter_rev}:{prompt_sha256(drafter_prompt)}"


def _json_line_map(raw_text: str) -> dict[tuple[str | int, ...], int]:
    """Walk valid JSON text and record 1-based line numbers for every object key and array element."""
    lines: dict[tuple[str | int, ...], int] = {}
    n = len(raw_text)

    def skip_ws(idx: int) -> int:
        while idx < n and raw_text[idx] in " \t\r\n":
            idx += 1
        return idx

    def line_at(idx: int) -> int:
        return raw_text.count("\n", 0, idx) + 1

    def parse_val(idx: int, path: tuple[str | int, ...]) -> int:
        idx = skip_ws(idx)
        if idx >= n:
            return idx
        lines.setdefault(path, line_at(idx))
        ch = raw_text[idx]
        if ch == "{":
            idx = skip_ws(idx + 1)
            if idx < n and raw_text[idx] == "}":
                return idx + 1
            while idx < n:
                idx = skip_ws(idx)
                key_line = line_at(idx)
                key, idx = json.decoder.scanstring(raw_text, idx + 1)
                sub_path = (*path, key)
                lines[sub_path] = key_line
                idx = skip_ws(idx)
                if idx < n and raw_text[idx] == ":":
                    idx += 1
                idx = parse_val(idx, sub_path)
                idx = skip_ws(idx)
                if idx < n and raw_text[idx] == ",":
                    idx += 1
                    continue
                if idx < n and raw_text[idx] == "}":
                    return idx + 1
                break
            return idx
        if ch == "[":
            idx = skip_ws(idx + 1)
            if idx < n and raw_text[idx] == "]":
                return idx + 1
            elem_i = 0
            while idx < n:
                idx = skip_ws(idx)
                elem_path = (*path, elem_i)
                lines[elem_path] = line_at(idx)
                idx = parse_val(idx, elem_path)
                elem_i += 1
                idx = skip_ws(idx)
                if idx < n and raw_text[idx] == ",":
                    idx += 1
                    continue
                if idx < n and raw_text[idx] == "]":
                    return idx + 1
                break
            return idx
        if ch == '"':
            _, next_idx = json.decoder.scanstring(raw_text, idx + 1)
            return next_idx
        m = _JSON_PRIM_RE.match(raw_text, idx)
        return m.end() if m else idx + 1

    parse_val(0, ())
    return lines


@dataclass(frozen=True)
class ClassCaptionRule:
    """Per-class required terms and banned words from project config, with 1-based line numbers."""

    name: str
    required_terms: tuple[str, ...]
    term_lines: tuple[int, ...]
    banned_words: tuple[str, ...]
    line: int = 1


@dataclass(frozen=True)
class CaptionRules:
    """Per-project caption rules loaded from `caption-rules.json` (or `--caption-config`)."""

    config_label: str
    prefix: str
    prefix_line: int
    drafter: str | None
    drafter_line: int
    drafter_prompt: str | None
    drafter_prompt_line: int
    class_field: str | None
    class_pattern: str | None
    compiled_pattern: re.Pattern[str] | None
    backdrop_text: str
    backdrop_field: str | None
    backdrop_required: bool
    backdrop_banned_words: tuple[str, ...]
    backdrop_line: int
    global_banned_words: tuple[str, ...]
    classes: dict[str, ClassCaptionRule]
    multimodal_pool: str | None = None
    multimodal_pool_line: int = 1

    def _lookup_dot_field(self, raw_record: dict, field_path: str | None) -> str:
        if not field_path or not isinstance(raw_record, dict):
            return ""
        cur: Any = raw_record
        for part in field_path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is None and field_path in raw_record:
            cur = raw_record[field_path]
        return cur.strip() if isinstance(cur, str) else ""

    def resolve_class(self, raw_record: dict, asset_path: str = "") -> str | None:
        if self.class_field:
            val = self._lookup_dot_field(raw_record, self.class_field)
            if val:
                return val
        if self.compiled_pattern is not None and asset_path:
            norm_asset = asset_path.replace("\\", "/")
            m = self.compiled_pattern.search(norm_asset) or self.compiled_pattern.search(os.path.basename(norm_asset))
            if m is not None:
                if "class" in m.groupdict() and m.group("class"):
                    return str(m.group("class")).strip()
                if m.lastindex:
                    return str(m.group(1)).strip()
                return str(m.group(0)).strip()
        return None

    def resolve_backdrop(
        self,
        raw_record: dict,
        record_file: str,
        *,
        record_writer: str | None = None,
    ) -> tuple[str, str | None]:
        if self.backdrop_field:
            rec_val = self._lookup_dot_field(raw_record, self.backdrop_field)
            if rec_val:
                return rec_val, (record_writer or f"record:{record_file}")
        if self.backdrop_text:
            return self.backdrop_text, f"config:{self.config_label}:{self.backdrop_line}"
        return "", None

    def required_terms_for_class(self, cls_name: str | None) -> list[tuple[str, int]]:
        if not cls_name or cls_name not in self.classes:
            return []
        rule = self.classes[cls_name]
        return list(zip(rule.required_terms, rule.term_lines, strict=True))

    def banned_words_for_class(self, cls_name: str | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        sources: list[tuple[str, ...]] = [self.global_banned_words, self.backdrop_banned_words]
        if cls_name and cls_name in self.classes:
            sources.append(self.classes[cls_name].banned_words)
        for seq in sources:
            for item in seq:
                key = item.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(item.strip())
        return out


def _config_display_label(config_path: str, project: str | None) -> str:
    abs_cfg = os.path.abspath(config_path)
    if project:
        abs_proj = os.path.abspath(project)
        try:
            if os.path.commonpath([abs_proj, abs_cfg]) == abs_proj:
                return os.path.relpath(abs_cfg, abs_proj).replace("\\", "/")
        except ValueError:
            pass
    return os.path.basename(abs_cfg)


def _as_str_tuple(val: Any, label: str) -> tuple[str, ...]:
    if val is None:
        return ()
    if not isinstance(val, list) or any(not isinstance(x, str) for x in val):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {label} must be a list of strings")
    return tuple(x.strip() for x in val if x.strip())


def _build_caption_rules(
    doc: dict,
    *,
    line_map: dict[tuple[str | int, ...], int] | None = None,
    config_label: str = CAPTION_RULES_FILENAME,
) -> CaptionRules:
    if not isinstance(doc, dict):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} must be a JSON object")
    lmap = line_map or {}

    pool_raw = doc.get("multimodal_pool")
    if pool_raw is not None and not isinstance(pool_raw, str):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'multimodal_pool' must be a string")
    multimodal_pool: str | None = None
    if isinstance(pool_raw, str) and pool_raw.strip():
        norm_pool = pool_raw.strip().lower()
        if norm_pool not in KNOWN_MULTIMODAL_POOLS:
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"caption config {config_label!r} has unknown 'multimodal_pool' {pool_raw!r}; "
                f"expected one of {sorted(KNOWN_MULTIMODAL_POOLS)}",
            )
        multimodal_pool = norm_pool
    multimodal_pool_line = lmap.get(("multimodal_pool",), 1)

    prefix_raw = doc.get("prefix")
    if prefix_raw is None and multimodal_pool is not None:
        prefix = ""
    else:
        if not isinstance(prefix_raw, str) or not prefix_raw.strip():
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"caption config {config_label!r} must define a non-empty string 'prefix'",
            )
        prefix = prefix_raw.strip()
    prefix_line = lmap.get(("prefix",), 1)

    drafter_raw = doc.get("drafter")
    if drafter_raw is not None and not isinstance(drafter_raw, str):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'drafter' must be a string")
    drafter = drafter_raw.strip() if isinstance(drafter_raw, str) and drafter_raw.strip() else None
    drafter_line = lmap.get(("drafter",), 1)

    prompt_raw = doc.get("drafter_prompt")
    if prompt_raw is not None and not isinstance(prompt_raw, str):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'drafter_prompt' must be a string")
    drafter_prompt = prompt_raw.strip() if isinstance(prompt_raw, str) and prompt_raw.strip() else None
    drafter_prompt_line = lmap.get(("drafter_prompt",), 1)

    class_field_raw = doc.get("class_field")
    if class_field_raw is not None and not isinstance(class_field_raw, str):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'class_field' must be a string")
    class_field = class_field_raw.strip() if isinstance(class_field_raw, str) and class_field_raw.strip() else None

    class_pat_raw = doc.get("class_pattern")
    if class_pat_raw is not None and not isinstance(class_pat_raw, str):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'class_pattern' must be a string")
    class_pattern = class_pat_raw if isinstance(class_pat_raw, str) and class_pat_raw else None
    compiled_pat: re.Pattern[str] | None = None
    if class_pattern is not None:
        try:
            compiled_pat = re.compile(class_pattern)
        except re.error as err:
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"caption config {config_label!r} has invalid 'class_pattern' regex: {err}",
                cause=err,
            ) from err

    b_raw = doc.get("backdrop")
    backdrop_text = ""
    backdrop_field: str | None = None
    backdrop_required = False
    backdrop_banned: tuple[str, ...] = ()
    backdrop_line = lmap.get(("backdrop",), 1)
    if isinstance(b_raw, str):
        backdrop_text = b_raw.strip()
    elif isinstance(b_raw, dict):
        bt = b_raw.get("text")
        if bt is not None and not isinstance(bt, str):
            raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'backdrop.text' must be a string")
        backdrop_text = bt.strip() if isinstance(bt, str) else ""
        if "text" in b_raw:
            backdrop_line = lmap.get(("backdrop", "text"), backdrop_line)
        bf = b_raw.get("field")
        if bf is not None and not isinstance(bf, str):
            raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'backdrop.field' must be a string")
        backdrop_field = bf.strip() if isinstance(bf, str) and bf.strip() else None
        backdrop_required = bool(b_raw.get("required", False))
        backdrop_banned = _as_str_tuple(b_raw.get("banned_words"), "'backdrop.banned_words'")
    elif b_raw is not None:
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'backdrop' must be a string or object")

    global_banned = _as_str_tuple(doc.get("banned_words"), "'banned_words'")

    classes_raw = doc.get("classes") or {}
    if not isinstance(classes_raw, dict):
        raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} 'classes' must be an object")
    classes: dict[str, ClassCaptionRule] = {}
    for cls_name, cls_body in classes_raw.items():
        if not isinstance(cls_body, dict):
            raise DeskError("INPUT_BAD_ARGUMENT", f"caption config {config_label!r} class {cls_name!r} must be an object")
        req_raw = cls_body.get("required_terms")
        req_list_raw = req_raw if req_raw is not None else []
        if not isinstance(req_list_raw, list) or any(not isinstance(x, str) for x in req_list_raw):
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"caption config {config_label!r} class {cls_name!r} 'required_terms' must be a list of strings",
            )
        cls_line = lmap.get(("classes", cls_name, "required_terms"), lmap.get(("classes", cls_name), 1))
        req_terms: list[str] = []
        term_lines: list[int] = []
        for idx, t in enumerate(req_list_raw):
            if t.strip():
                req_terms.append(t.strip())
                term_lines.append(lmap.get(("classes", cls_name, "required_terms", idx), cls_line))
        ban_terms = _as_str_tuple(cls_body.get("banned_words"), f"class {cls_name!r} 'banned_words'")
        classes[str(cls_name)] = ClassCaptionRule(
            name=str(cls_name),
            required_terms=tuple(req_terms),
            term_lines=tuple(term_lines),
            banned_words=ban_terms,
            line=cls_line,
        )

    return CaptionRules(
        config_label=config_label,
        prefix=prefix,
        prefix_line=prefix_line,
        drafter=drafter,
        drafter_line=drafter_line,
        drafter_prompt=drafter_prompt,
        drafter_prompt_line=drafter_prompt_line,
        class_field=class_field,
        class_pattern=class_pattern,
        compiled_pattern=compiled_pat,
        backdrop_text=backdrop_text,
        backdrop_field=backdrop_field,
        backdrop_required=backdrop_required,
        backdrop_banned_words=backdrop_banned,
        backdrop_line=backdrop_line,
        global_banned_words=global_banned,
        classes=classes,
        multimodal_pool=multimodal_pool,
        multimodal_pool_line=multimodal_pool_line,
    )


def load_caption_rules(project: str | None = None, config_path: str | None = None) -> CaptionRules | None:
    """Load per-project caption rules from `--caption-config` or `<project>/caption-rules.json` when present."""
    target: str | None = None
    explicit = config_path is not None and config_path != ""
    if explicit:
        target = os.path.abspath(str(config_path))
        if not os.path.isfile(target):
            raise DeskError("INPUT_BAD_ARGUMENT", f"caption config file not found: {target}")
    elif project:
        cand = os.path.join(project, CAPTION_RULES_FILENAME)
        if os.path.isfile(cand):
            target = cand
    if target is None:
        return None
    try:
        with open(target, encoding="utf-8") as f:
            raw_text = f.read()
        doc = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as err:
        raise DeskError("INPUT_BAD_ARGUMENT", f"cannot parse caption config {target!r}: {err}", cause=err) from err
    line_map = _json_line_map(raw_text)
    label = _config_display_label(target, project)
    return _build_caption_rules(doc, line_map=line_map, config_label=label)


def coerce_caption_rules(
    rules: CaptionRules | dict | str | None,
    *,
    project: str | None = None,
) -> CaptionRules | None:
    if rules is None or isinstance(rules, CaptionRules):
        return rules
    if isinstance(rules, str):
        return load_caption_rules(project=project, config_path=rules)
    if isinstance(rules, dict):
        return _build_caption_rules(rules, config_label=CAPTION_RULES_FILENAME)
    raise DeskError("INPUT_BAD_ARGUMENT", f"unsupported caption_rules type: {type(rules).__name__}")


def stopwords_sha256(path: str = STOPWORDS_PATH) -> str:
    """Return the SHA-256 hex digest of the committed stopword list (with LF line endings normalized)."""
    with open(path, "rb") as f:
        raw = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def load_stopwords(path: str = STOPWORDS_PATH) -> frozenset[str]:
    """Load the frozen stopword set from stopwords.txt."""
    with open(path, encoding="utf-8") as f:
        return frozenset(line.strip().lower() for line in f if line.strip() and not line.startswith("#"))


STOPWORDS: frozenset[str] = load_stopwords()
STOPWORDS_SHA256: str = stopwords_sha256()


def words(text: str | None) -> list[str]:
    """Pinned tokenization: re.findall(r'\\w+', s.lower())."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def content_words(text: str | None, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Words (preserving sequence and repeats) not in the frozen stopword list."""
    return [w for w in words(text) if w not in stopwords]


def normalize_text(text: str | None) -> str:
    """Pinned pool exact-duplicate key: ''.join(words(s))."""
    return "".join(words(text))


def word_set(text: str | None) -> set[str]:
    """Distinct words under pinned tokenization."""
    return set(words(text))


def content_word_set(text: str | None, stopwords: frozenset[str] = STOPWORDS) -> set[str]:
    """Distinct content words."""
    return set(content_words(text, stopwords))


def content_jaccard(
    a: str | set[str],
    b: str | set[str],
    stopwords: frozenset[str] = STOPWORDS,
) -> float:
    """Content-word Jaccard similarity in [0.0, 1.0]."""
    ca = content_word_set(a, stopwords) if isinstance(a, str) else a
    cb = content_word_set(b, stopwords) if isinstance(b, str) else b
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


# Alias kept for callers expecting jaccard()
jaccard = content_jaccard


def word_share(desc_words: list[str], prompt_words: list[str]) -> float:
    """Number of description words (counting repeats) that occur anywhere in the prompt, over description word count."""
    if not desc_words or not prompt_words:
        return 0.0
    pset = set(prompt_words)
    return sum(1 for w in desc_words if w in pset) / len(desc_words)


def max_contiguous_run(a: list[str], b: list[str]) -> int:
    """Length of the longest contiguous run of words shared between sequences a and b."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for wa in a:
        curr = [0] * (len(b) + 1)
        for j, wb in enumerate(b):
            if wa == wb:
                c = prev[j] + 1
                curr[j + 1] = c
                if c > best:
                    best = c
        prev = curr
    return best


def contains_word_sequence(haystack: list[str], needle: list[str]) -> bool:
    """Return True if needle (non-empty) appears as a contiguous slice inside haystack."""
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def link_target(path: str, start: str) -> str:
    """A Markdown link target for `path`, seen from a file in `start`: relative when it can be, else a file:// URI.

    On Windows a relative path can't cross drives (a project on E: and --out on C:), so the link falls back to the
    absolute file URI."""
    try:
        return os.path.relpath(path, start).replace("\\", "/").replace(" ", "%20")
    except ValueError:
        return pathlib.Path(os.path.abspath(path)).as_uri()


def wraps_canon_subject(description: str | None, canon_subject: str | None) -> bool:
    """Return True if description contains canon.subject as a contiguous run of words (4+ words)."""
    cw = words(canon_subject)
    if len(cw) < MIN_CANON_SUBJECT_WORDS:
        return False
    dw = words(description)
    if not dw:
        return False
    return contains_word_sequence(dw, cw)


def prompt_echo_stats(
    description: str | None,
    prompt: str | None,
    stopwords: frozenset[str] = STOPWORDS,
) -> dict:
    """Pinned prompt-echo statistics on all words, content words, and contiguous word runs."""
    dw = words(description)
    pw = words(prompt)
    dcw = [w for w in dw if w not in stopwords]
    pcw = [w for w in pw if w not in stopwords]

    equals_prompt = bool(dw and pw and dw == pw)
    inside_prompt = bool(dw and pw and not equals_prompt and contains_word_sequence(pw, dw))
    run = max_contiguous_run(dw, pw)
    all_share = round(word_share(dw, pw), 6)
    c_share = round(word_share(dcw, pcw), 6)

    return {
        "equals_prompt": equals_prompt,
        "inside_prompt": inside_prompt,
        "prompt_contiguous_run": run,
        "prompt_word_share": all_share,
        "prompt_content_word_share": c_share,
    }


def load_siglip2_tokenizer():
    """Lazily load the cached siglip2-so400m-patch14-384 tokenizer."""
    from transformers import AutoTokenizer
    from dataset_desk import backends

    with backends.transformers_warnings_off():  # the same harmless bos/eos config warning as the model's
        return AutoTokenizer.from_pretrained(
            backends.SIGLIP2_REPO,
            revision=backends.SIGLIP2_REVISION,
            local_files_only=True,
        )


def siglip2_token_count(s: str, tokenizer=None) -> int:
    """Pinned SigLIP2 token count: len(tokenizer(s.lower(), padding=False, truncation=False)['input_ids']) on the
    cached siglip2-so400m-patch14-384 tokenizer, special tokens included."""
    if tokenizer is None:
        tokenizer = load_siglip2_tokenizer()
    return len(tokenizer((s or "").lower(), padding=False, truncation=False)["input_ids"])


def siglip2_token_counts(texts: list[str], tokenizer=None) -> list[int]:
    """Compute SigLIP2 token counts for a list of descriptions, lazily importing the SigLIP2 tokenizer when needed."""
    if not texts:
        return []
    if tokenizer is None:
        tokenizer = load_siglip2_tokenizer()
    return [siglip2_token_count(s, tokenizer=tokenizer) for s in texts]


def _extract_record_fields(raw: dict) -> dict:
    """Support both raw style-dataset-lab records/*.json dicts and desk.py Item.record dicts.

    The benchmark and record description is `judgment.explanation` (or `canon.subject` only when `judgment.explanation`
    is absent, which flags `not_current_description` and `wraps_canon_subject`)."""
    if "judgment" in raw or "canon" in raw or "provenance" in raw:
        c = raw.get("canon") or {}
        j = raw.get("judgment") or {}
        prov = raw.get("provenance") or {}
        rid = raw.get("id")
        asset_path = (raw.get("asset_path") or "").replace("\\", "/")
        status = j.get("status")
        j_exp = (j.get("explanation") or "").strip()
        c_subj = (c.get("subject") or "").strip()
        prompt = (prov.get("prompt") or "").strip()
        existing_author_kind = prov.get("author_kind") or j.get("author_kind")
    else:
        rid = raw.get("id")
        asset_path = (raw.get("asset_path") or "").replace("\\", "/")
        status = raw.get("status")
        j_exp = (raw.get("judgment_note") or raw.get("judgment_explanation") or "").strip()
        c_subj = (raw.get("canon_subject") or "").strip()
        if not j_exp and not c_subj and raw.get("explanation"):
            j_exp = (raw.get("explanation") or "").strip()
        prompt = (raw.get("prompt") or "").strip()
        existing_author_kind = raw.get("author_kind")

    sidecar_file = (raw.get("_sidecar_file") or raw.get("sidecar_file") or "").replace("\\", "/")
    if not sidecar_file and "provenance" in raw:
        prov_src = (raw.get("provenance") or {}).get("source") or ""
        if isinstance(prov_src, str) and prov_src.startswith("sidecar:"):
            sidecar_file = prov_src[len("sidecar:"):].replace("\\", "/")

    if sidecar_file:
        text = j_exp if status == "approved" else ""
        source = f"sidecar:{sidecar_file}"
        record_file = sidecar_file
        record_writer = f"sidecar:{sidecar_file}"
    else:
        if j_exp and status == "approved":
            text = j_exp
            source = "judgment.explanation"
        elif c_subj:
            text = c_subj
            source = "canon.subject"
        else:
            text = ""
            source = "none"
        record_file = (
            raw.get("_record_file") or (f"records/{rid}.json" if rid else "records/unknown.json")
        ).replace("\\", "/")
        record_writer = f"record:{record_file}"

    return {
        "id": rid,
        "asset_path": asset_path,
        "status": status,
        "judgment_explanation": j_exp,
        "canon_subject": c_subj,
        "text": text,
        "source": source,
        "prompt": prompt,
        "existing_author_kind": existing_author_kind,
        "raw_record": raw,
        "record_file": record_file,
        "sidecar_file": sidecar_file or None,
        "record_writer": record_writer,
    }


def _connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return [g for g in groups.values() if len(g) > 1]


def audit_records(
    records: list[dict],
    queries: dict[str, str] | None = None,
    *,
    token_counter: Callable[[list[str]], list[int]] | dict[str, int] | None = None,
    stopwords: frozenset[str] = STOPWORDS,
    caption_rules: CaptionRules | dict | None = None,
    allow_composed_source: bool = False,
) -> dict:
    """Run Revision 3's acceptance tests (plus optional per-project caption rules) over a project's records.

    When `token_counter` is None, the stdlib acceptance tests run and records without model-free failures are marked
    `gate_status="unchecked"` and `passed=False` (because a record whose SigLIP2 token count was not computed cannot
    be marked as passing).
    """
    rules = coerce_caption_rules(caption_rules)
    extracted = [_extract_record_fields(r) for r in records]
    if queries is not None:
        by_id = {e["id"]: e for e in extracted if e.get("id")}
        target_list = []
        for rid, qtext in queries.items():
            base = dict(by_id.get(rid) or {
                "id": rid, "asset_path": "", "status": "missing",
                "judgment_explanation": "", "canon_subject": "", "prompt": "",
                "existing_author_kind": None, "raw_record": {}, "record_file": f"records/{rid}.json",
                "sidecar_file": None, "record_writer": f"record:records/{rid}.json",
            })
            qt = (qtext or "").strip()
            base["text"] = qt
            if allow_composed_source and qt:
                base["source"] = "composed"
            elif qt and qt == base["judgment_explanation"]:
                base["source"] = (
                    f"sidecar:{base['sidecar_file']}" if base.get("sidecar_file") else "judgment.explanation"
                )
            elif qt and qt == base["canon_subject"]:
                base["source"] = "canon.subject"
            elif not qt:
                base["source"] = f"sidecar:{base['sidecar_file']}" if base.get("sidecar_file") else "none"
            else:
                base["source"] = "custom"
            target_list.append(base)
    else:
        target_list = [e for e in extracted if e["status"] == "approved"]

    # 1. Exact duplicate groups after punctuation is removed ("".join(words(s)))
    by_norm_text: dict[str, list[str]] = {}
    by_norm_jexp: dict[str, list[str]] = {}
    raw_text_for_norm: dict[str, str] = {}
    for e in target_list:
        rid = str(e["id"])
        nt = normalize_text(e["text"])
        if nt:
            by_norm_text.setdefault(nt, []).append(rid)
            raw_text_for_norm.setdefault(nt, e["text"])
        if not allow_composed_source:
            nj = normalize_text(e["judgment_explanation"])
            if nj:
                by_norm_jexp.setdefault(nj, []).append(rid)
                raw_text_for_norm.setdefault(nj, e["judgment_explanation"])

    exact_groups: list[dict] = []
    seen_group_sigs: set[tuple[str, tuple[str, ...]]] = set()
    exact_member_ids: dict[str, list[str]] = {}
    exact_group_sizes: dict[str, int] = {}
    exact_keys: dict[str, set[tuple[str, str]]] = {}  # rid -> the exact groups it belongs to (pairs are implied)

    for label_source, bucket in (("active", by_norm_text), ("judgment.explanation", by_norm_jexp)):
        for norm_val, ids in bucket.items():
            if len(ids) > 1:
                sig = (norm_val, tuple(sorted(ids)))
                if sig not in seen_group_sigs:
                    seen_group_sigs.add(sig)
                    exact_groups.append({
                        "text": raw_text_for_norm[norm_val],
                        "normalized": norm_val,
                        "source": label_source,
                        "count": len(ids),
                        "ids": list(ids),
                    })
                for rid in ids:
                    exact_member_ids.setdefault(rid, []).append(raw_text_for_norm[norm_val])
                    exact_group_sizes[rid] = max(exact_group_sizes.get(rid, 1), len(ids))
                for rid in ids:
                    exact_keys.setdefault(rid, set()).add((label_source, norm_val))

    exact_groups.sort(key=lambda g: (-g["count"], g["normalized"]))

    # 2. Content-word Jaccard >= 0.85 among records in the pool
    cword_sets = {str(e["id"]): content_word_set(e["text"], stopwords) for e in target_list}
    near_pairs: list[tuple[str, str, float]] = []
    near_member_ids: set[str] = set()
    max_cjaccard_by_id: dict[str, float | None] = {str(e["id"]): None for e in target_list}
    ids_ordered = [str(e["id"]) for e in target_list]
    # Prefix filtering (Chaudhuri et al. 2006): order content words by increasing pool document frequency (ties broken
    # lexically) and index only each record's prefix of length L - ceil(tau * L) + 1. Any pair with Jaccard >= tau
    # shares at least one word in their prefixes and satisfies ceil(tau * |ca|) <= |cb| <= floor(|ca| / tau).
    tau = MAX_CONTENT_JACCARD - _EPS
    df: Counter[str] = Counter()
    for rid in ids_ordered:
        df.update(cword_sets[rid])
    prefixes: list[tuple[str, ...]] = []
    prefix_postings: dict[str, list[int]] = {}
    for j, rid in enumerate(ids_ordered):
        cw = cword_sets[rid]
        if not cw:
            prefixes.append(())
            continue
        ordered_cw = sorted(cw, key=lambda w: (df[w], w))
        pref_len = max(1, len(ordered_cw) - math.ceil(tau * len(ordered_cw)) + 1)
        pref = tuple(ordered_cw[:pref_len])
        prefixes.append(pref)
        for w in pref:
            prefix_postings.setdefault(w, []).append(j)
    for i in range(len(ids_ordered)):
        ia = ids_ordered[i]
        ca = cword_sets[ia]
        if not ca:
            continue
        len_a = len(ca)
        min_len_b = math.ceil(tau * len_a)
        max_len_b = math.floor(len_a / tau)
        cand_js = {
            j
            for w in prefixes[i]
            for j in prefix_postings[w][bisect.bisect_right(prefix_postings[w], i):]
            if min_len_b <= len(cword_sets[ids_ordered[j]]) <= max_len_b
        }
        keys_a = exact_keys.get(ia)
        for j in sorted(cand_js):
            ib = ids_ordered[j]
            if keys_a and keys_a & exact_keys.get(ib, set()):
                continue
            cb = cword_sets[ib]
            n_inter = len(ca & cb)
            sim = n_inter / (len_a + len(cb) - n_inter)
            if sim >= MAX_CONTENT_JACCARD - _EPS:
                r_sim = round(sim, 6)
                if max_cjaccard_by_id[ia] is None or r_sim > max_cjaccard_by_id[ia]:
                    max_cjaccard_by_id[ia] = r_sim
                if max_cjaccard_by_id[ib] is None or r_sim > max_cjaccard_by_id[ib]:
                    max_cjaccard_by_id[ib] = r_sim
                near_pairs.append((ia, ib, round(sim, 4)))
                near_member_ids.add(ia)
                near_member_ids.add(ib)

    near_groups = _connected_components(ids_ordered, [(a, b) for a, b, _ in near_pairs])

    # Optional lazy SigLIP2 token counting
    token_counts_by_id: dict[str, int] = {}
    if callable(token_counter):
        texts_to_count = [e["text"] for e in target_list]
        counts = token_counter(texts_to_count)
        for e, tc in zip(target_list, counts, strict=True):
            token_counts_by_id[str(e["id"])] = int(tc)
    elif isinstance(token_counter, dict):
        for e in target_list:
            rid = str(e["id"])
            if rid in token_counter:
                token_counts_by_id[rid] = int(token_counter[rid])
            elif e["text"] in token_counter:
                token_counts_by_id[rid] = int(token_counter[e["text"]])

    # 3. Per-record evaluation
    rows: list[dict] = []
    for e in target_list:
        rid = str(e["id"])
        flags: list[str] = []
        reasons: list[str] = []
        dup_size = exact_group_sizes.get(rid, 1)

        dw = words(e["text"])
        dcw = content_words(e["text"], stopwords)

        if e["status"] != "approved":
            flags.append("not_approved")
            reasons.append(f"status is {e['status']!r}, not 'approved'")

        if not dw:
            flags.append("empty_description")
            reasons.append("no approved description text")

        is_sidecar_src = isinstance(e["source"], str) and e["source"].startswith("sidecar:")
        if e["source"] not in ("judgment.explanation", "composed") and not is_sidecar_src:
            flags.append("not_current_description")
            if e["source"] == "canon.subject":
                reasons.append("text comes from canon.subject, not judgment.explanation")
            elif e["source"] == "custom":
                reasons.append("query text does not match record's current judgment.explanation")

        # Prompt echo tests (4 sub-tests)
        echo = prompt_echo_stats(e["text"], e["prompt"], stopwords)
        echo_reasons: list[str] = []
        if echo["equals_prompt"]:
            echo_reasons.append("equals generation prompt")
        elif echo["inside_prompt"]:
            echo_reasons.append("sits inside generation prompt")
        if echo["prompt_contiguous_run"] >= MAX_PROMPT_CONTIGUOUS_RUN:
            echo_reasons.append(
                f"shares contiguous run of {echo['prompt_contiguous_run']} >= {MAX_PROMPT_CONTIGUOUS_RUN} words with prompt"
            )
        if echo["prompt_word_share"] >= MAX_PROMPT_WORD_SHARE - _EPS:
            echo_reasons.append(
                f"prompt word share {echo['prompt_word_share']:.2f} >= {MAX_PROMPT_WORD_SHARE:.2f}"
            )
        if echo["prompt_content_word_share"] >= MAX_PROMPT_CONTENT_WORD_SHARE - _EPS:
            echo_reasons.append(
                f"prompt content-word share {echo['prompt_content_word_share']:.2f} >= {MAX_PROMPT_CONTENT_WORD_SHARE:.2f}"
            )
        if echo_reasons:
            flags.append("prompt_echo")
            reasons.extend(echo_reasons)

        # Wraps canon.subject (>= 4 words)
        wraps_canon = wraps_canon_subject(e["text"], e["canon_subject"])
        if wraps_canon:
            flags.append("wraps_canon_subject")
            reasons.append("contains canon.subject as a contiguous run of words (4+ words)")

        # Too short (< 8 words or < 5 content words)
        if len(dw) < MIN_WORDS or len(dcw) < MIN_CONTENT_WORDS:
            flags.append("too_short")
            reasons.append(
                f"too short ({len(dw)} words, {len(dcw)} content words; need >= {MIN_WORDS} words and >= {MIN_CONTENT_WORDS} content words)"
            )

        # Per-project model-free caption rules (prefix, class required terms, banned words)
        rec_class: str | None = None
        if rules is not None:
            rec_class = rules.resolve_class(e["raw_record"], e["asset_path"])
            pref_words = words(rules.prefix)
            if pref_words and dw[:len(pref_words)] != pref_words:
                flags.append("prefix_missing")
                reasons.append(f"missing required prefix {rules.prefix!r}")

            missing_terms: list[str] = []
            for req_term, _term_line in rules.required_terms_for_class(rec_class):
                tw = words(req_term)
                if tw and not contains_word_sequence(dw, tw):
                    missing_terms.append(req_term)
            if rules.backdrop_required:
                b_text, _b_writer = rules.resolve_backdrop(
                    e["raw_record"], e["record_file"], record_writer=e["record_writer"]
                )
                bw_req = words(b_text)
                if not bw_req:
                    missing_terms.append("<backdrop>")
                elif not contains_word_sequence(dw, bw_req):
                    missing_terms.append(b_text)
            if missing_terms:
                flags.append("required_term_missing")
                reasons.append(
                    f"missing required term(s) for class {rec_class or 'default'!r}: "
                    f"{', '.join(repr(t) for t in missing_terms)}"
                )

            hit_banned: list[str] = []
            for banned in rules.banned_words_for_class(rec_class):
                bw_seq = words(banned)
                if bw_seq and contains_word_sequence(dw, bw_seq):
                    hit_banned.append(banned)
            if hit_banned:
                flags.append("banned_word")
                reasons.append(
                    f"contains banned word(s) for class {rec_class or 'default'!r}: "
                    f"{', '.join(repr(b) for b in hit_banned)}"
                )

        # Pool duplicates (punctuation-stripped exact or content-word Jaccard >= 0.85)
        if rid in exact_member_ids:
            flags.append("exact_duplicate")
            reasons.append(f"exact duplicate description/ruling shared across {dup_size} records")

        if rid in near_member_ids:
            flags.append("near_duplicate")
            reasons.append(
                f"content-word Jaccard {max_cjaccard_by_id[rid]:.2f} >= {MAX_CONTENT_JACCARD:.2f} with another record"
            )

        # SigLIP2 token limit (> 64 tokens)
        tc = token_counts_by_id.get(rid)
        if tc is not None:
            token_status = "checked"
            if tc > MAX_SIGLIP2_TOKENS:
                flags.append("too_long_siglip2")
                reasons.append(f"SigLIP2 token count {tc} > {MAX_SIGLIP2_TOKENS}")
        else:
            token_status = "unchecked"

        passed_model_free = len(flags) == 0
        if len(flags) > 0:
            gate_status = "failed"
        elif token_status == "unchecked":
            gate_status = "unchecked"
        else:
            gate_status = "passed"

        # A record whose token count wasn't computed is "unchecked", and can't be marked as passing.
        passed = gate_status == "passed"
        known_strata = ("human", "captioner_assisted", "captioner_draft_unedited")
        author_kind = (
            e["existing_author_kind"]
            if (e["existing_author_kind"] in known_strata and passed_model_free)
            else ("human" if passed_model_free else "captioner_assisted")
        )

        init_writers = [e["record_writer"]] if e.get("sidecar_file") else None
        init_parts = (
            [{"part": "free", "text": e["text"], "writer": e["record_writer"]}]
            if (e.get("sidecar_file") and e["text"])
            else None
        )

        rows.append({
            "id": rid,
            "asset_path": e["asset_path"],
            "record_file": e["record_file"],
            "caption_class": rec_class,
            "status": e["status"],
            "source": e["source"],
            "text": e["text"],
            "judgment_explanation": e["judgment_explanation"],
            "canon_subject": e["canon_subject"],
            "prompt": e["prompt"],
            "word_count": len(dw),
            "content_word_count": len(dcw),
            "equals_prompt": echo["equals_prompt"],
            "inside_prompt": echo["inside_prompt"],
            "prompt_contiguous_run": echo["prompt_contiguous_run"],
            "prompt_word_share": echo["prompt_word_share"],
            "prompt_content_word_share": echo["prompt_content_word_share"],
            "wraps_canon_subject": wraps_canon,
            "max_content_jaccard": max_cjaccard_by_id.get(rid),
            "duplicate_group_size": dup_size,
            "siglip2_tokens": tc,
            "token_status": token_status,
            "flags": flags,
            "reasons": reasons,
            "passed_model_free": passed_model_free,
            "gate_status": gate_status,
            "passed": passed,
            "author_kind": author_kind,
            "draft": "",
            "florence2_draft": "",
            "drafter_id": None,
            "drafter_revision": None,
            "drafter_prompt": None,
            "drafter_prompt_sha256": None,
            "drafter_raw_output": None,
            "drafter_writer": None,
            "seat_edit": None,
            "seat_id": None,
            "writers": init_writers,
            "parts": init_parts,
            "final_caption": None,
        })

    flagged = [r for r in rows if r["gate_status"] == "failed"]
    passed_rows = [r for r in rows if r["passed"]]
    unchecked_rows = [r for r in rows if r["gate_status"] == "unchecked"]
    passed_mf_rows = [r for r in rows if r["passed_model_free"]]

    shares = sorted(r["prompt_word_share"] for r in rows)
    c_shares = sorted(r["prompt_content_word_share"] for r in rows)
    mid = len(shares) // 2
    median_share = (shares[mid] if len(shares) % 2 == 1 else 0.5 * (shares[mid - 1] + shares[mid])) if shares else 0.0
    median_c_share = (
        (c_shares[mid] if len(c_shares) % 2 == 1 else 0.5 * (c_shares[mid - 1] + c_shares[mid])) if c_shares else 0.0
    )
    max_exact = max((g["count"] for g in exact_groups), default=(1 if rows else 0))

    return {
        "passed_gate": len(rows) > 0 and len(passed_rows) == len(rows),
        "passed_model_free_gate": len(rows) > 0 and len(flagged) == 0,
        "stopwords_sha256": STOPWORDS_SHA256,
        "caption_config": rules.config_label if rules is not None else None,
        "checked": len(rows),
        "total_checked": len(rows),
        "passed_count": len(passed_rows),
        "passed_model_free_count": len(passed_mf_rows),
        "unchecked_count": len(unchecked_rows),
        "flagged_count": len(flagged),
        "max_exact_group": max_exact,
        "median_prompt_word_share": round(median_share, 4),
        "median_prompt_content_word_share": round(median_c_share, 4),
        "jaccard_threshold": MAX_CONTENT_JACCARD,
        "prompt_share_threshold": MAX_PROMPT_WORD_SHARE,
        "prompt_content_share_threshold": MAX_PROMPT_CONTENT_WORD_SHARE,
        "prompt_contiguous_run_threshold": MAX_PROMPT_CONTIGUOUS_RUN,
        "min_words": MIN_WORDS,
        "min_content_words": MIN_CONTENT_WORDS,
        "max_siglip2_tokens": MAX_SIGLIP2_TOKENS,
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "near_pairs": near_pairs,
        "records": rows,
        "flagged": flagged,
        "queue": flagged,
    }


def _trace_comfy_prompt_text(graph: dict, link: Any) -> str | None:
    visited: set[str] = set()
    cur = link
    while isinstance(cur, list) and cur and isinstance(cur[0], (str, int)):
        nid = str(cur[0])
        if nid in visited or nid not in graph:
            break
        visited.add(nid)
        node = graph[nid]
        if not isinstance(node, dict):
            break
        inp = node.get("inputs") or {}
        for key in ("text", "prompt", "text_g", "text_l"):
            val = inp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        cur = inp.get("conditioning") or inp.get("conditioning_1") or inp.get("positive")
    return None


def _extract_png_prompt_text(path: str) -> str:
    """Extract embedded ComfyUI PNG tEXt `prompt` string when present, returning '' otherwise."""
    if os.path.splitext(path)[1].lower() != ".png":
        return ""
    try:
        from PIL import Image

        with Image.open(path) as im:
            raw = (getattr(im, "info", None) or {}).get("prompt")
    except Exception:  # noqa: BLE001
        return ""
    if not raw or not isinstance(raw, str):
        return ""
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(graph, dict):
        return ""
    fallback_texts: list[str] = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type") or ""
        inp = node.get("inputs") or {}
        if cls in ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced", "RandomNoise"):
            pos_txt = _trace_comfy_prompt_text(graph, inp.get("positive"))
            if pos_txt:
                return pos_txt
        elif cls in ("CLIPTextEncode", "TextEncodeQwenImageEditPlus"):
            txt = (inp.get("text") or inp.get("prompt") or "").strip()
            if txt:
                fallback_texts.append(txt)
    return fallback_texts[0] if fallback_texts else ""


def _load_wave_prompts(project: str) -> dict[str, str]:
    out_dir = os.path.join(project, "outputs")
    if not os.path.isdir(out_dir):
        return {}
    prompts: dict[str, str] = {}
    for folder in sorted(os.listdir(out_dir)):
        gen_path = os.path.join(out_dir, folder, "generation.json")
        if not os.path.isfile(gen_path):
            continue
        try:
            with open(gen_path, encoding="utf-8") as f:
                gen = json.load(f)
            for item in gen.get("items") or []:
                fname = item.get("file")
                p_txt = (item.get("prompt") or "").strip()
                if fname and p_txt:
                    prompts[f"outputs/{folder}/{fname}".replace("\\", "/")] = p_txt
        except (OSError, json.JSONDecodeError):
            continue
    return prompts


def scan_sidecar_project(project: str) -> dict:
    """Scan `project` for images paired with same-stem `.txt` sidecars.

    - Each `(image, <stem>.txt)` pair becomes an approved record with provenance source `sidecar:<file>`.
    - Images without a sidecar are listed in `images_without_sidecar` and are never given invented text.
    - Nothing is written inside `project`.
    """
    abs_proj = os.path.abspath(project)
    if not os.path.isdir(abs_proj):
        raise DeskError("INPUT_BAD_ARGUMENT", f"project folder {project!r} does not exist")

    outputs_root = os.path.join(abs_proj, "outputs")
    skip_top_outputs = os.path.isdir(outputs_root) and any(
        os.path.isdir(os.path.join(outputs_root, d)) for d in os.listdir(outputs_root)
    )
    wave_prompts = _load_wave_prompts(abs_proj)

    raw_pairs: list[tuple[str, str, str, str, str]] = []  # (stem, rel_img, full_img, rel_txt, full_txt)
    images_without_sidecar: list[str] = []
    orphan_sidecars: list[str] = []

    for root, dirs, files in os.walk(abs_proj):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "records", "__pycache__"))
        is_top_outputs = skip_top_outputs and os.path.abspath(root) == os.path.abspath(outputs_root)
        if is_top_outputs:
            continue

        txt_by_lower_stem: dict[str, str] = {}
        txt_exact_stems: dict[str, str] = {}
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() == ".txt":
                txt_exact_stems.setdefault(stem, fn)
                txt_by_lower_stem.setdefault(stem.lower(), fn)

        paired_txt_names: set[str] = set()
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in SIDECAR_IMAGE_EXT:
                continue
            full_img = os.path.join(root, fn)
            rel_img = os.path.relpath(full_img, abs_proj).replace("\\", "/")
            txt_fn = txt_exact_stems.get(stem) or txt_by_lower_stem.get(stem.lower())
            if txt_fn is None:
                images_without_sidecar.append(rel_img)
                continue
            paired_txt_names.add(txt_fn)
            full_txt = os.path.join(root, txt_fn)
            rel_txt = os.path.relpath(full_txt, abs_proj).replace("\\", "/")
            raw_pairs.append((stem, rel_img, full_img, rel_txt, full_txt))

        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() == ".txt" and fn not in paired_txt_names:
                rel_orphan = os.path.relpath(os.path.join(root, fn), abs_proj).replace("\\", "/")
                orphan_sidecars.append(rel_orphan)

    stem_counts = Counter(stem for stem, *_rest in raw_pairs)
    records: list[dict] = []
    for stem, rel_img, full_img, rel_txt, full_txt in raw_pairs:
        rid = (
            stem
            if stem_counts[stem] == 1
            else os.path.splitext(rel_img)[0].replace("/", "__")
        )
        with open(full_txt, encoding="utf-8", errors="replace") as f:
            sidecar_text = f.read().strip()
        gen_prompt = wave_prompts.get(rel_img) or _extract_png_prompt_text(full_img)
        records.append({
            "id": rid,
            "asset_path": rel_img,
            "judgment": {
                "status": "approved",
                "explanation": sidecar_text,
            },
            "provenance": {
                "prompt": gen_prompt,
                "source": f"sidecar:{rel_txt}",
            },
            "_sidecar_file": rel_txt,
        })

    return {
        "records": records,
        "images_without_sidecar": images_without_sidecar,
        "orphan_sidecars": orphan_sidecars,
        "total_images": len(records) + len(images_without_sidecar),
    }


def format_pool_detection_note(pool_detection: dict | None) -> str:
    """Format a single-line Markdown summary of `pool_detection` with counts and source paths only (never hashes)."""
    if not isinstance(pool_detection, dict):
        return ""
    status = str(pool_detection.get("status") or "off").strip()
    if status == "on":
        pools_map = pool_detection.get("pools")
        counts_map = pool_detection.get("counts") or {}
        pool_names = sorted(
            pools_map.keys()
            if isinstance(pools_map, dict) and pools_map
            else (counts_map.keys() if isinstance(counts_map, dict) and counts_map else KNOWN_MULTIMODAL_POOLS)
        )
        segments: list[str] = []
        for p_name in pool_names:
            p_entry = pools_map.get(p_name) if isinstance(pools_map, dict) else None
            cnt = int(p_entry.get("count", 0)) if isinstance(p_entry, dict) else int(counts_map.get(p_name, 0))
            srcs = p_entry.get("sources") if isinstance(p_entry, dict) else None
            if isinstance(srcs, list) and srcs:
                src_labels = [
                    f"{s.get('via', 'source')}: {s.get('path', '')}".replace("\n", " ").strip()
                    for s in srcs
                    if isinstance(s, dict)
                ]
                segments.append(f"{p_name}: {cnt} ({'; '.join(src_labels)})" if src_labels else f"{p_name}: {cnt}")
            else:
                segments.append(f"{p_name}: {cnt}")
        return f"Pool detection (`pool_detection`): `on` — {', '.join(segments)}."
    warn_txt = str(pool_detection.get("warning") or "warning: renamed-copy detection is off").replace("\n", " ").strip()
    return f"Pool detection (`pool_detection`): `off` — {warn_txt}."


def save_sidecar_import(
    out_dir: str,
    scan_result: dict,
    *,
    pool_detection: dict | None = None,
) -> tuple[str, str]:
    """Persist imported sidecar records and missing-sidecar report under `out_dir` (never inside `project`)."""
    os.makedirs(out_dir, exist_ok=True)
    recs_path = os.path.join(out_dir, SIDECAR_RECORDS_FILENAME)
    with open(recs_path, "w", encoding="utf-8") as f:
        for rec in scan_result.get("records") or []:
            row_doc = {**rec, "pool_detection": pool_detection} if pool_detection is not None else rec
            f.write(json.dumps(row_doc, ensure_ascii=False) + "\n")
    rep_path = os.path.join(out_dir, SIDECAR_REPORT_FILENAME)
    rep_doc: dict[str, Any] = {
        "total_images": int(scan_result.get("total_images", 0)),
        "paired_count": len(scan_result.get("records") or []),
        "images_without_sidecar": list(scan_result.get("images_without_sidecar") or []),
        "orphan_sidecars": list(scan_result.get("orphan_sidecars") or []),
    }
    if pool_detection is not None:
        rep_doc["pool_detection"] = pool_detection
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return recs_path, rep_path


def load_sidecar_import(out_dir: str) -> tuple[list[dict], list[str]]:
    """Load cached sidecar records and missing-sidecar paths from `out_dir` when present."""
    recs_path = os.path.join(out_dir, SIDECAR_RECORDS_FILENAME)
    rep_path = os.path.join(out_dir, SIDECAR_REPORT_FILENAME)
    recs: list[dict] = []
    missing: list[str] = []
    if os.path.isfile(recs_path):
        with open(recs_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    if isinstance(doc, dict):
                        recs.append(doc)
    if os.path.isfile(rep_path):
        try:
            with open(rep_path, encoding="utf-8") as f:
                rep_doc = json.load(f)
            if isinstance(rep_doc, dict):
                missing = [str(p) for p in (rep_doc.get("images_without_sidecar") or [])]
        except (OSError, json.JSONDecodeError):
            missing = []
    return recs, missing


def import_sidecars(
    project: str,
    out_dir: str,
    *,
    caption_rules: CaptionRules | dict | str | None = None,
    drafter=None,
    dhashes: dict[str, str] | None = None,
    image_vecs: dict[str, Any] | None = None,
    token_counter: Callable[[list[str]], list[int]] | dict[str, int] | None = None,
    pool_detection: dict | None = None,
) -> dict:
    """Import `(image, .txt)` sidecar pairs from `project`, run the description gate, and write review files under `out_dir`."""
    _ensure_out_outside_project(project, out_dir)
    from . import desk

    if pool_detection is None:
        pool_detection = desk.check_multimodal_pool(project, "none", images_only=True)
    eff_rules = (
        coerce_caption_rules(caption_rules, project=project)
        if caption_rules is not None
        else load_caption_rules(project=project)
    )
    scan_res = scan_sidecar_project(project)
    recs_path, rep_path = save_sidecar_import(out_dir, scan_res, pool_detection=pool_detection)
    audit = audit_records(
        scan_res["records"],
        token_counter=token_counter,
        caption_rules=eff_rules,
    )
    audit["images_without_sidecar"] = list(scan_res["images_without_sidecar"])
    audit["pool_detection"] = pool_detection
    if drafter is not None:
        draft_flagged_records(project, audit, drafter)
    rev_jsonl, rev_ws = write_review_files(
        project,
        out_dir,
        audit,
        dhashes=dhashes,
        image_vecs=image_vecs,
        pool_detection=pool_detection,
    )
    desk.write_pool_detection_status(out_dir, pool_detection, command="sidecar-import")
    return {
        "total_images": scan_res["total_images"],
        "paired": len(scan_res["records"]),
        "images_without_sidecar": scan_res["images_without_sidecar"],
        "orphan_sidecars": scan_res["orphan_sidecars"],
        "passed_model_free_count": audit["passed_model_free_count"],
        "flagged_count": audit["flagged_count"],
        "sidecar_records_path": recs_path,
        "sidecar_report_path": rep_path,
        "review_jsonl_path": rev_jsonl,
        "worksheet_path": rev_ws,
        "pool_detection": pool_detection,
        "audit": audit,
    }


def load_project_records(project: str) -> list[dict]:
    """Read every JSON file under <project>/records/ using stdlib only."""
    rdir = os.path.join(project, "records")
    if not os.path.isdir(rdir):
        return []
    out = []
    for name in sorted(os.listdir(rdir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(rdir, name), encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict):
                doc.setdefault("_record_file", f"records/{name}")
                out.append(doc)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def audit_project(
    project: str,
    queries: dict[str, str] | None = None,
    *,
    token_counter: Callable[[list[str]], list[int]] | dict[str, int] | None = None,
    caption_rules: CaptionRules | dict | str | None = None,
) -> dict:
    """Audit all approved records (or a query manifest) in a style-dataset-lab project folder."""
    eff_rules = (
        coerce_caption_rules(caption_rules, project=project)
        if caption_rules is not None
        else load_caption_rules(project=project)
    )
    return audit_records(
        load_project_records(project),
        queries=queries,
        token_counter=token_counter,
        caption_rules=eff_rules,
    )


def draft_flagged_records(project: str, audit: dict, drafter) -> dict:
    """Ask plain-sight (Florence-2), Qwen3-VL, or a test drafter for a blind draft of each flagged record.

    Passes ONLY the image file path to the drafter — never the generation prompt, `canon.subject`, or the current text.
    Stores the prompt text, prompt SHA-256, and full raw output with each draft.
    """
    flagged = audit.get("flagged") or []
    drafter_id = getattr(drafter, "drafter_id", getattr(drafter, "name", "plain-sight"))
    drafter_rev = getattr(drafter, "revision", "unknown")
    d_prompt = getattr(drafter, "prompt", None)
    d_prompt_sha = prompt_sha256(d_prompt)
    if not flagged:
        audit["drafter_id"] = drafter_id
        audit["drafter_revision"] = drafter_rev
        audit["drafter_prompt"] = d_prompt
        audit["drafter_prompt_sha256"] = d_prompt_sha
        return audit

    image_paths = [
        os.path.join(project, r["asset_path"]) if r.get("asset_path") else ""
        for r in flagged
    ]
    drafts = drafter.draft(image_paths)
    # Re-read drafter_id and revision after lazy _ensure_model() inside draft()
    drafter_id = getattr(drafter, "drafter_id", getattr(drafter, "name", "plain-sight"))
    drafter_rev = getattr(drafter, "revision", "unknown")
    raw_map = getattr(drafter, "raw_outputs", None) or {}
    last_raw = getattr(drafter, "last_raw_outputs", None)
    d_writer = drafter_writer_id(drafter_id, drafter_rev, d_prompt)

    for idx, (r, img_p, dtext) in enumerate(zip(flagged, image_paths, drafts, strict=True)):
        clean = (dtext or "").strip()
        raw_out = (
            last_raw[idx]
            if (isinstance(last_raw, list) and idx < len(last_raw))
            else raw_map.get(img_p, dtext or "")
        )
        r["draft"] = clean
        r["florence2_draft"] = clean
        r["author_kind"] = "captioner_assisted"
        r["caption_stratum"] = "captioner_draft_unedited"
        r["drafter_id"] = drafter_id
        r["drafter_revision"] = drafter_rev
        r["drafter_prompt"] = d_prompt
        r["drafter_prompt_sha256"] = d_prompt_sha
        r["drafter_raw_output"] = raw_out
        r["drafter_writer"] = d_writer

    audit["drafter_id"] = drafter_id
    audit["drafter_revision"] = drafter_rev
    audit["drafter_prompt"] = d_prompt
    audit["drafter_prompt_sha256"] = d_prompt_sha
    return audit


def write_drafter_raw(
    out_dir: str,
    rows: list[dict],
    *,
    pool_detection: dict | None = None,
) -> str | None:
    """Persist per-image full raw drafter outputs under `<out_dir>/drafter-raw.jsonl` when present."""
    drafted = [r for r in rows if r.get("drafter_id") and (r.get("drafter_raw_output") is not None or r.get("draft") or r.get("florence2_draft"))]
    if not drafted:
        return None
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, DRAFTER_RAW_FILENAME)
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in drafted:
            row_doc: dict[str, Any] = {
                "id": r["id"],
                "asset_path": r.get("asset_path", ""),
                "drafter_id": r.get("drafter_id"),
                "drafter_revision": r.get("drafter_revision"),
                "drafter_prompt": r.get("drafter_prompt"),
                "drafter_prompt_sha256": r.get("drafter_prompt_sha256"),
                "drafter_writer": r.get("drafter_writer"),
                "raw_output": (
                    r["drafter_raw_output"]
                    if r.get("drafter_raw_output") is not None
                    else (r.get("draft") or r.get("florence2_draft") or "")
                ),
                "draft": r.get("draft") or r.get("florence2_draft") or "",
            }
            if pool_detection is not None:
                row_doc["pool_detection"] = pool_detection
            f.write(json.dumps(row_doc, ensure_ascii=False) + "\n")
    return raw_path


def _hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _fallback_hash_hex(asset_path: str) -> str:
    return hashlib.sha256(asset_path.encode("utf-8")).hexdigest()


def _dot_cosine(va: Any, vb: Any) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(va, vb, strict=True)))


def select_siblings(
    target_asset: str,
    pool_assets: list[dict],
    k_min: int = 3,
    k_max: int = 5,
    *,
    image_vecs: dict[str, Any] | None = None,
) -> list[str]:
    """Choose 3 to 5 sibling images from the same project folder, nearest by DINOv2 image-only cosine similarity when
    `image_vecs` is provided and otherwise nearest by dHash (falling back to other folders if fewer than `k_min`)."""
    target_norm = (target_asset or "").replace("\\", "/")
    target_folder = os.path.dirname(target_norm)
    target_dh = None
    for item in pool_assets:
        if item["asset_path"] == target_norm:
            target_dh = item.get("dhash")
            break
    if not target_dh:
        target_dh = _fallback_hash_hex(target_norm)

    norm_vecs: dict[str, Any] = {}
    if image_vecs:
        for k, v in image_vecs.items():
            norm_vecs[k.replace("\\", "/")] = v
    target_vec = norm_vecs.get(target_norm)

    same_folder: list[tuple[tuple[float, int, str], str]] = []
    other_folder: list[tuple[tuple[float, int, str], str]] = []
    seen_paths: set[str] = {target_norm}

    for item in pool_assets:
        cand_path = item["asset_path"]
        if not cand_path or cand_path in seen_paths:
            continue
        seen_paths.add(cand_path)
        cand_dh = item.get("dhash") or _fallback_hash_hex(cand_path)
        dist = _hamming_hex(target_dh, cand_dh)
        cand_vec = norm_vecs.get(cand_path) if target_vec is not None else None
        neg_cos = -round(_dot_cosine(target_vec, cand_vec), 6) if cand_vec is not None else 2.0
        sort_key = (neg_cos, dist, cand_path)
        if os.path.dirname(cand_path) == target_folder:
            same_folder.append((sort_key, cand_path))
        else:
            other_folder.append((sort_key, cand_path))

    same_folder.sort(key=lambda t: t[0])
    other_folder.sort(key=lambda t: t[0])

    chosen = [p for _, p in same_folder[:k_max]]
    if len(chosen) < k_min and other_folder:
        need = k_max - len(chosen)
        chosen.extend(p for _, p in other_folder[:need])
    return chosen[:k_max]


def _escape_md_cell(val: str | None) -> str:
    return (val or "").replace("|", "\\|").replace("\n", " ").strip()


def write_review_files(
    project: str,
    out_dir: str,
    audit: dict,
    *,
    dhashes: dict[str, str] | None = None,
    image_vecs: dict[str, Any] | None = None,
    seat_edits: dict[str, dict] | None = None,
    pool_detection: dict | None = None,
) -> tuple[str, str]:
    """Write description-review.jsonl and description-worksheet.md under out_dir.

    The blind worksheet shows for every flagged record:
    - the id;
    - the target image;
    - 3 to 5 sibling images from dHash / DINOv2 image-only similarity (never text embeddings in a locked pool);
    - the instruction;
    - author_kind;
    - the drafter's draft;
    - a column for the seat's edited text and seat_id.

    The worksheet NEVER shows the current text, `canon.subject`, the drafter prompt, or the generation prompt.
    """
    os.makedirs(out_dir, exist_ok=True)
    eff_pool_detection = pool_detection if pool_detection is not None else audit.get("pool_detection")
    dh_map = dict(dhashes or {})
    edits = seat_edits if seat_edits is not None else load_seat_edits(out_dir)
    pool_assets: list[dict] = []
    seen_pool: set[str] = set()
    for path_key, dh_val in sorted(dh_map.items()):
        norm_p = path_key.replace("\\", "/")
        if norm_p not in seen_pool:
            seen_pool.add(norm_p)
            pool_assets.append({"asset_path": norm_p, "dhash": dh_val})
    for r in audit["records"]:
        ap = (r.get("asset_path") or "").replace("\\", "/")
        if ap and ap not in seen_pool:
            seen_pool.add(ap)
            pool_assets.append({"asset_path": ap, "dhash": dh_map.get(ap) or _fallback_hash_hex(ap)})

    write_drafter_raw(out_dir, audit["records"], pool_detection=eff_pool_detection)

    jsonl_path = os.path.join(out_dir, "description-review.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in audit["records"]:
            sibs = (
                select_siblings(r["asset_path"], pool_assets, image_vecs=image_vecs)
                if not r["passed_model_free"]
                else []
            )
            ed_entry = edits.get(str(r["id"]))
            seat_edit_val = r.get("seat_edit") or (ed_entry["edited_text"] if ed_entry else None)
            seat_id_val = r.get("seat_id") or (ed_entry["seat"] if ed_entry else None)
            draft_val = r.get("draft") or r.get("florence2_draft") or ""
            row_out: dict[str, Any] = {
                "id": r["id"],
                "asset_path": r["asset_path"],
                "caption_class": r.get("caption_class"),
                "status": r["status"],
                "source": r["source"],
                "gate_status": r["gate_status"],
                "passed_model_free": r["passed_model_free"],
                "passed": r["passed"],
                "author_kind": r["author_kind"],
                "drafter_id": r["drafter_id"],
                "drafter_revision": r["drafter_revision"],
                "drafter_prompt": r.get("drafter_prompt"),
                "drafter_prompt_sha256": r.get("drafter_prompt_sha256"),
                "drafter_raw_output": r.get("drafter_raw_output"),
                "drafter_writer": r.get("drafter_writer"),
                "draft": draft_val,
                "florence2_draft": r.get("florence2_draft") or draft_val,
                "seat_edit": seat_edit_val,
                "seat_id": seat_id_val,
                "writers": r.get("writers"),
                "parts": r.get("parts"),
                "final_caption": r.get("final_caption"),
                "siblings": sibs,
                "flags": r["flags"],
                "reasons": r["reasons"],
                "word_count": r["word_count"],
                "content_word_count": r["content_word_count"],
                "prompt_word_share": r["prompt_word_share"],
                "prompt_content_word_share": r["prompt_content_word_share"],
                "prompt_contiguous_run": r["prompt_contiguous_run"],
                "wraps_canon_subject": r["wraps_canon_subject"],
                "max_content_jaccard": r["max_content_jaccard"],
                "duplicate_group_size": r["duplicate_group_size"],
                "siglip2_tokens": r["siglip2_tokens"],
                "stopwords_sha256": audit["stopwords_sha256"],
                "current_text": r["text"] or r["judgment_explanation"],
                "canon_subject": r["canon_subject"],
                "prompt": r["prompt"],
            }
            if eff_pool_detection is not None:
                row_out["pool_detection"] = eff_pool_detection
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    def _fmt_link(asset: str) -> str:
        if not asset:
            return "(no image path)"
        return f"[{asset}]({link_target(os.path.join(project, asset), out_dir)})"

    sib_mode = "dHash/DINOv2 image-only" if image_vecs else "dHash"
    ws_path = os.path.join(out_dir, "description-worksheet.md")
    lines = [
        f"# Re-description worksheet: {os.path.basename(os.path.abspath(project))}",
        "",
        f"**Instruction:** {WORKSHEET_INSTRUCTION}",
        "",
        f"Stopwords sha256: `{audit['stopwords_sha256']}`.",
    ]
    if eff_pool_detection is not None:
        lines.append(format_pool_detection_note(eff_pool_detection))
    if audit.get("drafter_id"):
        lines.append(f"Drafter: `{audit['drafter_id']}` @ `{audit.get('drafter_revision', 'unknown')}`.")
    lines += [
        "",
        f"| id | target image | sibling images (nearest by {sib_mode}) | instruction | author_kind | Florence-2 draft | edited text | seat_id |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in audit["flagged"]:
        target_link = _fmt_link(r["asset_path"])
        sib_paths = select_siblings(r["asset_path"], pool_assets, image_vecs=image_vecs)
        sib_links = ", ".join(_fmt_link(sp) for sp in sib_paths) if sib_paths else "(none)"
        draft_clean = _escape_md_cell(r.get("draft") or r.get("florence2_draft"))
        ed_entry = edits.get(str(r["id"]))
        seat_edit_clean = _escape_md_cell(r.get("seat_edit") or (ed_entry["edited_text"] if ed_entry else ""))
        seat_id_clean = _escape_md_cell(r.get("seat_id") or (ed_entry["seat"] if ed_entry else ""))
        lines.append(
            f"| `{r['id']}` | {target_link} | {sib_links} | {WORKSHEET_INSTRUCTION} | "
            f"`{r['author_kind']}` | {draft_clean} | {seat_edit_clean} | {seat_id_clean} |"
        )

    human_rows = [r for r in audit["records"] if r["passed_model_free"]]
    if human_rows:
        lines += [
            "",
            "## Existing descriptions passing model-free gate (`author_kind: human`)",
            "",
            "| id | target image | gate_status | author_kind |",
            "|---|---|---|---|",
        ]
        for r in human_rows:
            lines.append(
                f"| `{r['id']}` | {_fmt_link(r['asset_path'])} | `{r['gate_status']}` | `{r['author_kind']}` |"
            )

    missing_sc = list(audit.get("images_without_sidecar") or [])
    if missing_sc:
        lines += [
            "",
            "## Images without `.txt` sidecar (reported, never invented)",
            "",
        ]
        for rel_img in missing_sc:
            lines.append(f"- {_fmt_link(rel_img)}")

    with open(ws_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return jsonl_path, ws_path


def _ensure_out_outside_project(project: str | None, out_dir: str) -> None:
    if not project:
        return
    abs_proj = os.path.abspath(project)
    abs_out = os.path.abspath(out_dir)
    try:
        inside = os.path.commonpath([abs_proj, abs_out]) == abs_proj
    except ValueError:
        inside = False
    if inside:
        raise DeskError(
            "PERM_OUT_INSIDE_PROJECT",
            "REFUSED: --out is inside the project; the desk never writes into a project",
        )


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_seat_edits(out_dir: str) -> dict[str, dict]:
    """Load previously imported seat edits from `<out_dir>/seat-edits.jsonl`."""
    p = os.path.join(out_dir, SEAT_EDITS_FILENAME)
    if not os.path.isfile(p):
        return {}
    by_id: dict[str, dict] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("id") and row.get("edited_text"):
                by_id[str(row["id"])] = row
    return by_id


def _split_md_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [c.replace("\\|", "|").strip() for c in _UNESCAPED_PIPE_RE.split(stripped)]


def import_worksheet_edits(
    worksheet_path: str,
    out_dir: str,
    *,
    seat: str | None = None,
    project: str | None = None,
    pool_detection: dict | None = None,
) -> dict:
    """Re-import blind worksheet edits under `out_dir` and record each edit with its `seat:<name>` writer id."""
    _ensure_out_outside_project(project, out_dir)
    if not os.path.isfile(worksheet_path):
        raise DeskError("INPUT_BAD_ARGUMENT", f"worksheet file not found: {worksheet_path}")
    default_seat = (seat or "").strip()
    if default_seat and not _SEAT_NAME_RE.match(default_seat):
        raise DeskError("INPUT_BAD_ARGUMENT", f"invalid seat name {default_seat!r}: use letters, digits, '.', '_' or '-'")
    from . import desk

    eff_pd = pool_detection
    if eff_pd is None:
        eff_pd = (
            desk.check_multimodal_pool(project, "none", images_only=True)
            if project
            else desk.resolve_pool_detection(emit_warning=True)[0]
        )

    ws_sha = _sha256_file(worksheet_path)
    with open(worksheet_path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    header_cols: list[str] | None = None
    id_idx = -1
    edit_idx = -1
    seat_idx = -1
    imported_rows: list[dict] = []

    for line in raw_lines:
        s = line.strip()
        if not s.startswith("|"):
            if header_cols is not None and s.startswith("## "):
                break
            continue
        cells = _split_md_table_row(s)
        lower_cells = [c.lower() for c in cells]
        if header_cols is None:
            if "id" in lower_cells and "edited text" in lower_cells:
                header_cols = lower_cells
                id_idx = lower_cells.index("id")
                edit_idx = lower_cells.index("edited text")
                seat_idx = lower_cells.index("seat_id") if "seat_id" in lower_cells else -1
            continue
        if all(set(c) <= {"-", ":"} for c in cells if c):
            continue
        if len(cells) <= max(id_idx, edit_idx):
            continue
        rid = cells[id_idx].strip().strip("`").strip()
        edit_text = cells[edit_idx].strip()
        if not rid or not edit_text:
            continue
        row_seat = cells[seat_idx].strip().strip("`").strip() if (seat_idx >= 0 and seat_idx < len(cells)) else ""
        eff_seat = row_seat or default_seat
        if not eff_seat:
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"edited row {rid!r} has no seat id; pass --seat <name> or fill the seat_id column",
            )
        if not _SEAT_NAME_RE.match(eff_seat):
            raise DeskError("INPUT_BAD_ARGUMENT", f"invalid seat name {eff_seat!r} on row {rid!r}")
        imported_rows.append({
            "id": rid,
            "edited_text": edit_text,
            "seat": eff_seat,
            "writer": f"seat:{eff_seat}",
            "worksheet_sha256": ws_sha,
            "pool_detection": eff_pd,
        })

    os.makedirs(out_dir, exist_ok=True)
    merged = load_seat_edits(out_dir)
    for row in imported_rows:
        merged[row["id"]] = row
    for existing_row in merged.values():
        existing_row["pool_detection"] = eff_pd

    edits_path = os.path.join(out_dir, SEAT_EDITS_FILENAME)
    with open(edits_path, "w", encoding="utf-8") as f:
        for rid_key in sorted(merged):
            f.write(json.dumps(merged[rid_key], ensure_ascii=False) + "\n")

    rev_path = os.path.join(out_dir, "description-review.jsonl")
    if os.path.isfile(rev_path):
        updated_rev: list[dict] = []
        with open(rev_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    updated_rev.append(json.loads(line))
        for r in updated_rev:
            ed = merged.get(str(r.get("id")))
            r["pool_detection"] = eff_pd
            if ed is not None:
                r["seat_edit"] = ed["edited_text"]
                r["seat_id"] = ed["seat"]
                r["author_kind"] = "captioner_assisted" if r.get("drafter_id") else "human"
                existing_writers = list(r.get("writers") or [])
                if existing_writers:
                    existing_writers[-1] = ed["writer"]
                    r["writers"] = existing_writers
                else:
                    r["writers"] = [ed["writer"]]
            elif r.get("drafter_id") and not r.get("seat_edit"):
                r["author_kind"] = "captioner_draft_unedited"
        with open(rev_path, "w", encoding="utf-8") as f:
            for r in updated_rev:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    desk.write_pool_detection_status(out_dir, eff_pd, command="import-edits")
    return {
        "imported": len(imported_rows),
        "total_seat_edits": len(merged),
        "seat_edits_path": edits_path,
        "records": imported_rows,
        "pool_detection": eff_pd,
    }


def _join_caption_parts(texts: list[str]) -> str:
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return ""
    acc = cleaned[0]
    for nxt in cleaned[1:]:
        if acc.endswith((",", ";", ":", "—", "-")):
            acc = f"{acc} {nxt}"
        else:
            acc = f"{acc}, {nxt}"
    return acc


def _strip_leading_segments(text: str, leading_parts: list[str]) -> str:
    rem = (text or "").strip()
    for part in leading_parts:
        p_clean = (part or "").strip().rstrip(",;:")
        if not p_clean:
            continue
        pat = re.compile(r"^\s*" + re.escape(p_clean) + r"\s*[,;:]?\s*", flags=re.IGNORECASE)
        nxt = pat.sub("", rem, count=1)
        if nxt != rem:
            rem = nxt.strip()
    return rem


def _load_cached_review_rows(out_dir: str) -> dict[str, dict]:
    rev_path = os.path.join(out_dir, "description-review.jsonl")
    if not os.path.isfile(rev_path):
        return {}
    by_id: dict[str, dict] = {}
    with open(rev_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict) and row.get("id"):
                    by_id[str(row["id"])] = row
    return by_id


def compose_captions(
    project: str,
    out_dir: str,
    *,
    caption_rules: CaptionRules | dict | str | None = None,
    drafter=None,
    seat_edits: dict[str, dict] | None = None,
    dhashes: dict[str, str] | None = None,
    image_vecs: dict[str, Any] | None = None,
    token_counter: Callable[[list[str]], list[int]] | dict[str, int] | None = None,
    manifest_name: str = CAPTION_MANIFEST_FILENAME,
    sidecars: bool = False,
    pool_detection: dict | None = None,
) -> dict:
    """Compose final training captions under `out_dir` (never inside `project`) and emit a lane-3 dataset manifest.

    Each final caption is assembled from:
    1. `prefix` verbatim from project config (`config:<file>:<line>`);
    2. `class` required terms verbatim from project config (`config:<file>:<line>`);
    3. optional `backdrop` slot (`record:<file>`, `sidecar:<file>`, or `config:<file>:<line>`);
    4. free visible-content part from `seat:<name>` edit if one exists, else `drafter:<id>@<rev>:<prompt sha>`, else
       the existing approved record/sidecar description (`record:<file>` or `sidecar:<file>`).

    Edited drafter captions carry `author_kind: "captioner_assisted"`; unedited drafter captions carry
    `author_kind: "captioner_draft_unedited"`.
    """
    _ensure_out_outside_project(project, out_dir)
    rules = (
        coerce_caption_rules(caption_rules, project=project)
        if caption_rules is not None
        else load_caption_rules(project=project)
    )
    if rules is None or not rules.prefix:
        raise DeskError(
            "INPUT_BAD_ARGUMENT",
            f"compose requires a non-empty 'prefix' in {CAPTION_RULES_FILENAME} or --caption-config <path>",
        )
    from . import desk

    eff_pd = pool_detection if pool_detection is not None else desk.check_multimodal_pool(project, "none", images_only=True)

    raw_recs = load_project_records(project) if not sidecars else []
    missing_sidecars: list[str] = []
    if not raw_recs:
        cached_sc_recs, cached_missing = load_sidecar_import(out_dir)
        if sidecars or not cached_sc_recs:
            sc_scan = scan_sidecar_project(project)
            if sidecars or sc_scan["records"]:
                save_sidecar_import(out_dir, sc_scan, pool_detection=eff_pd)
                raw_recs = sc_scan["records"]
                missing_sidecars = sc_scan["images_without_sidecar"]
        else:
            raw_recs = cached_sc_recs
            missing_sidecars = cached_missing

    extracted = [e for e in (_extract_record_fields(r) for r in raw_recs) if e["status"] == "approved"]
    if not extracted:
        raise DeskError("INPUT_BAD_ARGUMENT", f"no approved records found in {project!r}")

    edits_map = load_seat_edits(out_dir)
    if seat_edits:
        edits_map.update(seat_edits)
    cached_rev = _load_cached_review_rows(out_dir)

    live_drafts: dict[str, dict] = {}
    if drafter is not None:
        todo_entries = [e for e in extracted if str(e["id"]) not in edits_map]
        if todo_entries:
            todo_paths = [os.path.join(project, e["asset_path"]) for e in todo_entries]
            for e, p in zip(todo_entries, todo_paths, strict=True):
                if not e["asset_path"] or not os.path.isfile(p):
                    raise DeskError("INPUT_BAD_ARGUMENT", f"record {e['id']!r} missing image file: {e['asset_path']!r}")
            draft_texts = drafter.draft(todo_paths)
            d_id = getattr(drafter, "drafter_id", getattr(drafter, "name", "plain-sight"))
            d_rev = getattr(drafter, "revision", "unknown")
            d_prompt = getattr(drafter, "prompt", None)
            d_psha = prompt_sha256(d_prompt)
            d_writer = drafter_writer_id(d_id, d_rev, d_prompt)
            raw_map = getattr(drafter, "raw_outputs", None) or {}
            last_raw = getattr(drafter, "last_raw_outputs", None)
            for idx, (e, img_p, dtext) in enumerate(zip(todo_entries, todo_paths, draft_texts, strict=True)):
                raw_out = (
                    last_raw[idx]
                    if (isinstance(last_raw, list) and idx < len(last_raw))
                    else raw_map.get(img_p, dtext or "")
                )
                live_drafts[str(e["id"])] = {
                    "draft": (dtext or "").strip(),
                    "drafter_id": d_id,
                    "drafter_revision": d_rev,
                    "drafter_prompt": d_prompt,
                    "drafter_prompt_sha256": d_psha,
                    "drafter_raw_output": raw_out,
                    "drafter_writer": d_writer,
                }

    seen_img_names: set[str] = set()
    seen_stems: set[str] = set()
    composed_by_id: dict[str, str] = {}
    meta_by_id: dict[str, dict] = {}
    manifest_items: list[dict] = []

    for e in extracted:
        rid = str(e["id"])
        asset_path = e["asset_path"]
        img_abs = os.path.join(project, asset_path) if asset_path else ""
        if not asset_path or not os.path.isfile(img_abs):
            raise DeskError("INPUT_BAD_ARGUMENT", f"record {rid!r} missing image file: {asset_path!r}")
        img_name = os.path.basename(asset_path)
        stem, suffix = os.path.splitext(img_name)
        if not _SAFE_IMAGE_NAME_RE.match(img_name) or not suffix or suffix.lower() == ".txt":
            raise DeskError("INPUT_BAD_ARGUMENT", f"record {rid!r} has invalid image filename {img_name!r}")
        if img_name in seen_img_names or stem in seen_stems:
            raise DeskError("INPUT_BAD_ARGUMENT", f"duplicate image filename or caption stem {img_name!r}")
        seen_img_names.add(img_name)
        seen_stems.add(stem)
        img_sha = _sha256_file(img_abs)

        parts: list[dict] = []
        leading_texts: list[str] = []

        # 1. Required prefix verbatim from config
        parts.append({
            "part": "prefix",
            "text": rules.prefix,
            "writer": f"config:{rules.config_label}:{rules.prefix_line}",
        })
        leading_texts.append(rules.prefix)

        # 2. Per-class required terms verbatim from config
        rec_class = rules.resolve_class(e["raw_record"], asset_path)
        for req_term, term_line in rules.required_terms_for_class(rec_class):
            parts.append({
                "part": "class",
                "text": req_term,
                "writer": f"config:{rules.config_label}:{term_line}",
            })
            leading_texts.append(req_term)

        # 3. Optional backdrop slot
        b_text, b_writer = rules.resolve_backdrop(
            e["raw_record"], e["record_file"], record_writer=e["record_writer"]
        )
        if b_text and b_writer:
            parts.append({
                "part": "backdrop",
                "text": b_text,
                "writer": b_writer,
            })
            leading_texts.append(b_text)

        # 4. Free visible-content part: seat edit > drafter > existing record/sidecar
        ed = edits_map.get(rid)
        d_info = live_drafts.get(rid)
        if d_info is None:
            c_row = cached_rev.get(rid)
            if c_row and c_row.get("drafter_id") and (c_row.get("draft") or c_row.get("florence2_draft")):
                c_prompt = c_row.get("drafter_prompt")
                c_id = str(c_row["drafter_id"])
                c_rev = str(c_row.get("drafter_revision") or "unknown")
                d_info = {
                    "draft": (c_row.get("draft") or c_row.get("florence2_draft") or "").strip(),
                    "drafter_id": c_id,
                    "drafter_revision": c_rev,
                    "drafter_prompt": c_prompt,
                    "drafter_prompt_sha256": c_row.get("drafter_prompt_sha256") or prompt_sha256(c_prompt),
                    "drafter_raw_output": c_row.get("drafter_raw_output"),
                    "drafter_writer": c_row.get("drafter_writer") or drafter_writer_id(c_id, c_rev, c_prompt),
                }

        if ed is not None:
            free_clean = _strip_leading_segments(ed["edited_text"], leading_texts)
            seat_writer = f"seat:{ed['seat']}"
            free_part: dict[str, Any] = {
                "part": "free",
                "text": free_clean,
                "writer": seat_writer,
            }
            if d_info is not None:
                free_part["draft_writer"] = d_info["drafter_writer"]
                stratum = "captioner_assisted"
            else:
                stratum = "human"
            parts.append(free_part)
        elif d_info is not None:
            free_clean = _strip_leading_segments(d_info["draft"], leading_texts)
            parts.append({
                "part": "free",
                "text": free_clean,
                "writer": d_info["drafter_writer"],
            })
            stratum = "captioner_draft_unedited"
        else:
            free_clean = _strip_leading_segments(e["text"], leading_texts)
            if free_clean:
                parts.append({
                    "part": "free",
                    "text": free_clean,
                    "writer": e["record_writer"],
                })
            stratum = (
                e["existing_author_kind"]
                if e["existing_author_kind"] in ("captioner_assisted", "captioner_draft_unedited")
                else "human"
            )

        final_caption = _join_caption_parts([p["text"] for p in parts])
        writers = [str(p["writer"]) for p in parts]
        composed_by_id[rid] = final_caption
        meta_by_id[rid] = {
            "author_kind": stratum,
            "source": e["record_writer"],
            "writers": writers,
            "parts": parts,
            "final_caption": final_caption,
            "draft_info": d_info,
            "seat_edit": ed["edited_text"] if ed else None,
            "seat_id": ed["seat"] if ed else None,
        }
        manifest_items.append({
            "id": rid,
            "image": img_name,
            "asset_path": asset_path,
            "sha256": img_sha,
            "caption": final_caption,
            "source": e["record_writer"],
            "author_kind": stratum,
            "writers": writers,
            "parts": parts,
            "drafter_writer": d_info["drafter_writer"] if d_info else None,
        })

    audit = audit_records(
        raw_recs,
        queries=composed_by_id,
        token_counter=token_counter,
        caption_rules=rules,
        allow_composed_source=True,
    )
    audit["pool_detection"] = eff_pd
    if missing_sidecars:
        audit["images_without_sidecar"] = list(missing_sidecars)
    first_d_info = next((m["draft_info"] for m in meta_by_id.values() if m["draft_info"] is not None), None)
    if first_d_info is not None:
        audit["drafter_id"] = first_d_info["drafter_id"]
        audit["drafter_revision"] = first_d_info["drafter_revision"]
        audit["drafter_prompt"] = first_d_info["drafter_prompt"]
        audit["drafter_prompt_sha256"] = first_d_info["drafter_prompt_sha256"]

    for r in audit["records"]:
        m = meta_by_id.get(str(r["id"]))
        if m is None:
            continue
        r["author_kind"] = m["author_kind"]
        r["writers"] = m["writers"]
        r["parts"] = m["parts"]
        r["final_caption"] = m["final_caption"]
        r["seat_edit"] = m["seat_edit"]
        r["seat_id"] = m["seat_id"]
        di = m["draft_info"]
        if di is not None:
            r["draft"] = di["draft"]
            r["florence2_draft"] = di["draft"]
            r["drafter_id"] = di["drafter_id"]
            r["drafter_revision"] = di["drafter_revision"]
            r["drafter_prompt"] = di["drafter_prompt"]
            r["drafter_prompt_sha256"] = di["drafter_prompt_sha256"]
            r["drafter_raw_output"] = di["drafter_raw_output"]
            r["drafter_writer"] = di["drafter_writer"]

    rev_jsonl, rev_ws = write_review_files(
        project,
        out_dir,
        audit,
        dhashes=dhashes,
        image_vecs=image_vecs,
        seat_edits=edits_map,
        pool_detection=eff_pd,
    )

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, manifest_name)
    manifest_doc = {"pool_detection": eff_pd, "items": manifest_items}
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    desk.write_pool_detection_status(out_dir, eff_pd, command="compose")

    return {
        "manifest_path": manifest_path,
        "review_jsonl_path": rev_jsonl,
        "worksheet_path": rev_ws,
        "items": manifest_items,
        "images_without_sidecar": missing_sidecars,
        "pool_detection": eff_pd,
        "audit": audit,
    }
