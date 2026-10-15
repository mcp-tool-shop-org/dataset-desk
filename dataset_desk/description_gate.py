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

import hashlib
import json
import os
import re
from typing import Callable

STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")

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
_EPS = 1e-12


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

    if j_exp and status == "approved":
        text = j_exp
        source = "judgment.explanation"
    elif c_subj:
        text = c_subj
        source = "canon.subject"
    else:
        text = ""
        source = "none"

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
) -> dict:
    """Run Revision 3's acceptance tests over a project's records (or a candidate query manifest).

    When `token_counter` is None, the stdlib acceptance tests run and records without model-free failures are marked
    `gate_status="unchecked"` and `passed=False` (because a record whose SigLIP2 token count was not computed cannot
    be marked as passing).
    """
    extracted = [_extract_record_fields(r) for r in records]
    if queries is not None:
        by_id = {e["id"]: e for e in extracted if e.get("id")}
        target_list = []
        for rid, qtext in queries.items():
            base = dict(by_id.get(rid) or {
                "id": rid, "asset_path": "", "status": "missing",
                "judgment_explanation": "", "canon_subject": "", "prompt": "",
                "existing_author_kind": None,
            })
            qt = (qtext or "").strip()
            base["text"] = qt
            if qt and qt == base["judgment_explanation"]:
                base["source"] = "judgment.explanation"
            elif qt and qt == base["canon_subject"]:
                base["source"] = "canon.subject"
            elif not qt:
                base["source"] = "none"
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
        nj = normalize_text(e["judgment_explanation"])
        if nj:
            by_norm_jexp.setdefault(nj, []).append(rid)
            raw_text_for_norm.setdefault(nj, e["judgment_explanation"])

    exact_groups: list[dict] = []
    seen_group_sigs: set[tuple[str, tuple[str, ...]]] = set()
    exact_member_ids: dict[str, list[str]] = {}
    exact_group_sizes: dict[str, int] = {}
    same_exact_pair: set[frozenset[str]] = set()

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
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        same_exact_pair.add(frozenset((ids[i], ids[j])))

    exact_groups.sort(key=lambda g: (-g["count"], g["normalized"]))

    # 2. Content-word Jaccard >= 0.85 among records in the pool
    cword_sets = {str(e["id"]): content_word_set(e["text"], stopwords) for e in target_list}
    near_pairs: list[tuple[str, str, float]] = []
    near_member_ids: set[str] = set()
    max_cjaccard_by_id: dict[str, float] = {str(e["id"]): 0.0 for e in target_list}
    ids_ordered = [str(e["id"]) for e in target_list]
    for i in range(len(ids_ordered)):
        ia = ids_ordered[i]
        ca = cword_sets[ia]
        if not ca:
            continue
        for j in range(i + 1, len(ids_ordered)):
            ib = ids_ordered[j]
            if frozenset((ia, ib)) in same_exact_pair:
                continue
            cb = cword_sets[ib]
            if not cb:
                continue
            sim = len(ca & cb) / len(ca | cb)
            if sim > max_cjaccard_by_id[ia]:
                max_cjaccard_by_id[ia] = round(sim, 6)
            if sim > max_cjaccard_by_id[ib]:
                max_cjaccard_by_id[ib] = round(sim, 6)
            if sim >= MAX_CONTENT_JACCARD - _EPS:
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

        if e["source"] != "judgment.explanation":
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
        author_kind = (
            e["existing_author_kind"]
            if (e["existing_author_kind"] in ("human", "captioner_assisted") and passed_model_free)
            else ("human" if passed_model_free else "captioner_assisted")
        )

        rows.append({
            "id": rid,
            "asset_path": e["asset_path"],
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
            "max_content_jaccard": max_cjaccard_by_id.get(rid, 0.0),
            "duplicate_group_size": dup_size,
            "siglip2_tokens": tc,
            "token_status": token_status,
            "flags": flags,
            "reasons": reasons,
            "passed_model_free": passed_model_free,
            "gate_status": gate_status,
            "passed": passed,
            "author_kind": author_kind,
            "florence2_draft": "",
            "drafter_id": None,
            "drafter_revision": None,
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
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def audit_project(
    project: str,
    queries: dict[str, str] | None = None,
    *,
    token_counter: Callable[[list[str]], list[int]] | dict[str, int] | None = None,
) -> dict:
    """Audit all approved records (or a query manifest) in a style-dataset-lab project folder."""
    return audit_records(
        load_project_records(project),
        queries=queries,
        token_counter=token_counter,
    )


def draft_flagged_records(project: str, audit: dict, drafter) -> dict:
    """Ask plain-sight (Florence-2) or a test drafter for a blind draft of each flagged record.

    Passes ONLY the image file path to the drafter — never the generation prompt, `canon.subject`, or the current text.
    Records each draft with `author_kind: "captioner_assisted"`, the drafter's id, and its pinned revision.
    """
    flagged = audit.get("flagged") or []
    if not flagged:
        audit["drafter_id"] = getattr(drafter, "drafter_id", None)
        audit["drafter_revision"] = getattr(drafter, "revision", None)
        return audit

    image_paths = [
        os.path.join(project, r["asset_path"]) if r.get("asset_path") else ""
        for r in flagged
    ]
    drafts = drafter.draft(image_paths)
    drafter_id = getattr(drafter, "drafter_id", getattr(drafter, "name", "plain-sight"))
    drafter_rev = getattr(drafter, "revision", "unknown")

    for r, dtext in zip(flagged, drafts, strict=True):
        r["florence2_draft"] = (dtext or "").strip()
        r["author_kind"] = "captioner_assisted"
        r["drafter_id"] = drafter_id
        r["drafter_revision"] = drafter_rev

    audit["drafter_id"] = drafter_id
    audit["drafter_revision"] = drafter_rev
    return audit


def _hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _fallback_hash_hex(asset_path: str) -> str:
    return hashlib.sha256(asset_path.encode("utf-8")).hexdigest()


def select_siblings(
    target_asset: str,
    pool_assets: list[dict],
    k_min: int = 3,
    k_max: int = 5,
) -> list[str]:
    """Choose 3 to 5 sibling images model-free: from the same project folder, nearest by dHash (falling back to other
    folders in the project only if the same folder has fewer than k_min candidates)."""
    target_norm = (target_asset or "").replace("\\", "/")
    target_folder = os.path.dirname(target_norm)
    target_dh = None
    for item in pool_assets:
        if item["asset_path"] == target_norm:
            target_dh = item.get("dhash")
            break
    if not target_dh:
        target_dh = _fallback_hash_hex(target_norm)

    same_folder: list[tuple[int, str]] = []
    other_folder: list[tuple[int, str]] = []
    seen_paths: set[str] = {target_norm}

    for item in pool_assets:
        cand_path = item["asset_path"]
        if not cand_path or cand_path in seen_paths:
            continue
        seen_paths.add(cand_path)
        cand_dh = item.get("dhash") or _fallback_hash_hex(cand_path)
        dist = _hamming_hex(target_dh, cand_dh)
        if os.path.dirname(cand_path) == target_folder:
            same_folder.append((dist, cand_path))
        else:
            other_folder.append((dist, cand_path))

    same_folder.sort(key=lambda t: (t[0], t[1]))
    other_folder.sort(key=lambda t: (t[0], t[1]))

    chosen = [p for _, p in same_folder[:k_max]]
    if len(chosen) < k_min and other_folder:
        need = k_max - len(chosen)
        chosen.extend(p for _, p in other_folder[:need])
    return chosen[:k_max]


def write_review_files(
    project: str,
    out_dir: str,
    audit: dict,
    *,
    dhashes: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Write description-review.jsonl and description-worksheet.md under out_dir.

    The worksheet shows for every flagged record:
    - the id;
    - the target image;
    - 3 to 5 sibling images, chosen model-free: from the same project folder, nearest by dHash;
    - the instruction: "describe this picture in 15–35 words so a person could pick it out from the siblings shown";
    - author_kind (captioner_assisted for drafted/flagged records; human for passing existing records);
    - a column for the Florence-2 draft;
    - a column for the edited text.

    The worksheet NEVER shows the current text, `canon.subject`, or the generation prompt.
    """
    os.makedirs(out_dir, exist_ok=True)
    dh_map = dict(dhashes or {})
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

    jsonl_path = os.path.join(out_dir, "description-review.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in audit["records"]:
            sibs = select_siblings(r["asset_path"], pool_assets) if not r["passed_model_free"] else []
            f.write(json.dumps({
                "id": r["id"],
                "asset_path": r["asset_path"],
                "status": r["status"],
                "source": r["source"],
                "gate_status": r["gate_status"],
                "passed_model_free": r["passed_model_free"],
                "passed": r["passed"],
                "author_kind": r["author_kind"],
                "drafter_id": r["drafter_id"],
                "drafter_revision": r["drafter_revision"],
                "florence2_draft": r["florence2_draft"],
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
            }, ensure_ascii=False) + "\n")

    def _fmt_link(asset: str) -> str:
        if not asset:
            return "(no image path)"
        rel = os.path.relpath(os.path.join(project, asset), out_dir).replace("\\", "/")
        return f"[{asset}]({rel.replace(' ', '%20')})"

    ws_path = os.path.join(out_dir, "description-worksheet.md")
    lines = [
        f"# Re-description worksheet: {os.path.basename(os.path.abspath(project))}",
        "",
        f"**Instruction:** {WORKSHEET_INSTRUCTION}",
        "",
        f"Stopwords sha256: `{audit['stopwords_sha256']}`.",
    ]
    if audit.get("drafter_id"):
        lines.append(f"Drafter: `{audit['drafter_id']}` @ `{audit.get('drafter_revision', 'unknown')}`.")
    lines += [
        "",
        "| id | target image | sibling images (nearest by dHash) | instruction | author_kind | Florence-2 draft | edited text |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in audit["flagged"]:
        target_link = _fmt_link(r["asset_path"])
        sib_paths = select_siblings(r["asset_path"], pool_assets)
        sib_links = ", ".join(_fmt_link(sp) for sp in sib_paths) if sib_paths else "(none)"
        draft_clean = (r.get("florence2_draft") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{r['id']}` | {target_link} | {sib_links} | {WORKSHEET_INSTRUCTION} | "
            f"`{r['author_kind']}` | {draft_clean} |  |"
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

    with open(ws_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return jsonl_path, ws_path
