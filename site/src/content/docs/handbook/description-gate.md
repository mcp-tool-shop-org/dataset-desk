---
title: The description gate
description: Every test the model-free description gate runs, with its exact arithmetic.
sidebar:
  order: 3
---

The gate decides whether an approved record's description can be trusted as a description of its image. It runs with
no model, so its verdict can't be shaped by a model that is later measured on the same text.

## The pinned arithmetic

- **Words** are `re.findall(r"\w+", s.lower())`.
- **A content word** is a word not in the frozen stopword list (`dataset_desk/stopwords.txt`, 212 words). Its sha256 is
  exposed as `description_gate.STOPWORDS_SHA256` and written into every audit, so two runs can show they used the same
  list.
- **A share** is the number of the description's words (counting repeats) that occur anywhere in the prompt, divided by
  the description's word count. The content-word share is the same over content words.
- **Contiguous runs** and the subject-line slice are matched on these word sequences.

## The tests

A description fails if any of these hold.

| Test | Fails when | Why |
|---|---|---|
| Equals or inside the prompt | its words equal the prompt's, or appear as a run inside them | it is the prompt |
| Shared run | it shares a contiguous run of **6 or more** words with the prompt | it copies a clause of the prompt |
| All-word share | **0.90 or more** of its words appear in the prompt | it is the prompt reworded |
| Content-word share | **0.80 or more** of its content words appear in the prompt | the same, ignoring filler |
| Wrapped subject | the record's `canon.subject` has **4 or more** words and they appear as a contiguous slice of its words | it is the subject line with padding |
| Too short | under **8** words, or under **5** content words | too little to pick the image out |
| Exact duplicate | `"".join(words(a)) == "".join(words(b))` for another description in the set | the same text on two images |
| Near duplicate | a content-word Jaccard of **0.85 or more** with another description | nearly the same text |
| Too long for SigLIP2 | more than **64** tokens | SigLIP2 would truncate it |

Notes:
- "Exact duplicate" joins words with no separator, so `cobble-stone arch-way` matches `cobblestone archway`.
- The duplicate tests compare every pair across the whole set, not within a folder.

## The token count

The SigLIP2 test is the one that needs a library. It counts

```python
len(tokenizer(s.lower(), padding=False, truncation=False)["input_ids"])
```

on the `google/siglip2-so400m-patch14-384` tokenizer at a pinned revision, special tokens included. Counting through the
processor's defaults would pad and truncate to 64, so a description could never fail.

The tokenizer loads lazily. When it isn't available, the record is **unchecked**, and an unchecked record is never
marked as passing.

## Reading `description-review.jsonl`

Each line is one approved record:

- `gate_status`: `passed`, `failed` or `unchecked`;
- `flags` and `reasons`: which tests failed, in words;
- the statistics behind them: the shares, the longest shared run, the largest Jaccard, the token count;
- `author_kind`: `human` for existing text, `captioner_assisted` for a drafted one.

## What the gate can't see

It is lexical. A description that paraphrases its prompt in different words passes. After a benchmark's counted run,
the embedding fit check in the report (does the description find its own image?) covers that case.
