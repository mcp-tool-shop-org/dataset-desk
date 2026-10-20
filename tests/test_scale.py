"""The many-image paths give the same results as the per-pair code they replaced, across block boundaries.
Run from the repository root: python -W error -m unittest discover -s tests"""

import json
import os
import tempfile
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from dataset_desk import description_gate, desk


def rand_hash(rng):
    return "".join(f"{int(x):016x}" for x in rng.integers(0, 2**63, size=4, dtype=np.int64))


def flip_bits(h, rng, k):
    """h with k random bits flipped: a near copy at Hamming distance k."""
    v = int(h, 16)
    for b in rng.choice(256, size=k, replace=False):
        v ^= 1 << int(b)
    return f"{v:064x}"


def make_items(n, seed=0, drop_hash=(), drop_flip=()):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        if i % 7 == 3 and i > 0:  # near copies of an earlier item, some through its mirror hash
            base = items[i - 3]
            src = base.dhash_flip if (i % 2 and base.dhash_flip) else base.dhash
            h = flip_bits(src or rand_hash(rng), rng, int(rng.integers(0, 40)))
        else:
            h = rand_hash(rng)
        items.append(desk.Item(
            sha256=f"{i:064x}", paths=[f"outputs/w/{i}.png"], folders=["w"],
            dhash="" if i in drop_hash else h,
            dhash_flip="" if i in drop_flip else rand_hash(rng),
            record={"status": "approved" if i % 3 else "rejected", "id": str(i)},
            palette=list(rng.random(30))))
    return items


def near_reference(items, near):
    """The per-pair loop that dupes() used before the many-image path."""
    out = []
    for a in range(len(items)):
        if not items[a].dhash:
            continue
        for b in range(a + 1, len(items)):
            if not items[b].dhash:
                continue
            h = desk.flip_hamming(items[a], items[b])
            if h <= near:
                out.append((a, b, h))
    return out


def outliers_reference(items, img, k=5, n=15):
    A = desk.approved_idx(items)
    S = img[A] @ img[A].T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(A) - 1)
    fit = np.sort(S, axis=1)[:, -kk:].mean(axis=1)
    least = [(A[i], float(fit[i])) for i in np.argsort(fit)[:n]]
    others = [i for i in range(len(items)) if items[i].status != "approved"]
    best = (img[others] @ img[A].T).max(axis=1)
    order = np.argsort(best)
    return {"approved": len(A), "k": kk, "least_typical": least,
            "nearest_canon": [(others[i], float(best[i])) for i in order[::-1][:n]],
            "furthest_from_canon": [(others[i], float(best[i])) for i in order[:n]]}


def look_fit_reference(items, k=5):
    A = [i for i in desk.approved_idx(items) if items[i].palette]
    X = np.array([items[i].palette for i in A], dtype=np.float32)
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 0.02)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    kk = min(k, len(A) - 1)
    return {A[i]: float(v) for i, v in enumerate(np.sort(D, axis=1)[:, :kk].mean(axis=1))}


def captions_reference(items, img, txt, n=15):
    """The per-description loop that captions() used before the blocked helper."""
    if img.ndim != 2 or img.shape[1] == 0:
        return {"described": 0, "pool": len(items), "recall_at_1": None, "recall_at_5": None,
                "median_rank": None, "group_recall_at_1": None, "group_median_rank": None, "worst": []}
    pos = {it.sha256: i for i, it in enumerate(items)}
    norm_desc = {
        i: ((it.record or {}).get("explanation") or "").strip().lower()
        for i, it in enumerate(items)
        if it.status == "approved"
    }
    desc_groups: dict[str, list[int]] = {}
    for i, d in norm_desc.items():
        if d:
            desc_groups.setdefault(d, []).append(i)

    rows = []
    for sha, v in txt.items():
        if sha not in pos:
            continue
        i = pos[sha]
        if items[i].status != "approved" or not (items[i].record or {}).get("explanation"):
            continue
        sims = img @ v
        rk = desk.mid_rank(sims, i)
        siblings = desc_groups.get(norm_desc.get(i, ""), [i])
        if len(siblings) > 1:
            masked = sims.copy()
            for sib in siblings:
                if sib != i:
                    masked[sib] = -np.inf
            grk = desk.mid_rank(masked, i)
        else:
            grk = rk
        rows.append({"item": i, "rank": rk, "group_rank": grk, "siblings": len(siblings), "cosine": float(sims[i])})
    rows.sort(key=lambda r: (-r["group_rank"], -r["rank"]))
    ranks = [r["rank"] for r in rows]
    granks = [r["group_rank"] for r in rows]
    return {"described": len(rows), "pool": len(items),
            "recall_at_1": sum(r <= 1 for r in ranks) / len(ranks) if ranks else None,
            "recall_at_5": sum(r <= 5 for r in ranks) / len(ranks) if ranks else None,
            "median_rank": float(np.median(ranks)) if ranks else None,
            "group_recall_at_1": sum(r <= 1 for r in granks) / len(granks) if granks else None,
            "group_median_rank": float(np.median(granks)) if granks else None,
            "worst": rows[:n]}


def swap_check_reference(items, img, txt, seed=0):
    """The per-description loop and pairwise AUC matrix that swap_check() used before the blocked helper."""
    if img.ndim != 2 or img.shape[1] == 0:
        return None
    pos = {it.sha256: i for i, it in enumerate(items)}
    shas = [
        sh for sh in sorted(txt)
        if sh in pos and items[pos[sh]].status == "approved" and (items[pos[sh]].record or {}).get("explanation")
    ]
    if len(shas) < 3:
        return None
    descs = [((items[pos[sh]].record or {}).get("explanation") or sh).strip().lower() for sh in shas]
    counts: dict[str, int] = {}
    for d in descs:
        counts[d] = counts.get(d, 0) + 1
    if len(counts) < 3:
        return None
    if max(counts.values()) > len(shas) // 2:
        seen_d = set()
        dedup_shas, dedup_descs = [], []
        for sh, d in zip(shas, descs, strict=True):
            if d not in seen_d:
                seen_d.add(d)
                dedup_shas.append(sh)
                dedup_descs.append(d)
        shas, descs = dedup_shas, dedup_descs
        counts = {d: 1 for d in descs}

    rng = np.random.default_rng(seed)
    n = len(shas)
    desc_arr = np.array(descs)
    perm = None
    for _ in range(64):
        cand = rng.permutation(n)
        if not np.any(cand == np.arange(n)) and not np.any(desc_arr[cand] == desc_arr):
            perm = cand
            break
    if perm is None:
        max_c = max(counts.values())
        group_keys = list(counts.keys())
        rng.shuffle(group_keys)
        g_rank = {g: idx for idx, g in enumerate(group_keys)}
        order = sorted(range(n), key=lambda i: (g_rank[descs[i]], float(rng.random())))
        shifted = np.roll(order, max_c)
        perm = np.empty(n, dtype=int)
        for src, dst in zip(order, shifted, strict=True):
            perm[src] = dst

    true = [desk.mid_rank(img @ txt[sh], pos[sh]) for sh in shas]
    swapped = [desk.mid_rank(img @ txt[shas[perm[k]]], pos[sh]) for k, sh in enumerate(shas)]
    t, w = np.array(true), np.array(swapped)
    auc = float(((w[:, None] > t[None, :]).sum() + 0.5 * (w[:, None] == t[None, :]).sum()) / (len(w) * len(t)))
    return {"pairs": len(shas), "auc": auc, "true_median_rank": float(np.median(t)),
            "swapped_median_rank": float(np.median(w)), "seed": seed}


def unit_vectors(n, d=32, seed=1):
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lineage_020(items, vecs, k=5):
    """0.2.0's lineage (d3fa354a), verbatim: every pair in Python, then sort and slice."""
    has_vecs = vecs.ndim == 2 and vecs.shape[1] > 0
    by_name: dict[str, list[dict]] = {}
    for idx, it in enumerate(items):
        occs = it.occurrences or [
            {"path": p, "run_id": it.run_id, "name": os.path.basename(p), "suffix": it.suffix, "outcome": it.outcome}
            for p in it.paths
        ]
        for occ in occs:
            ext = (occ.get("suffix") or os.path.splitext(occ.get("name") or "")[1]).lower()
            if ext not in desk.CODE_EXT:
                continue
            fname = occ.get("name") or os.path.basename(occ["path"])
            by_name.setdefault(fname, []).append({**occ, "item_idx": idx})

    files_out: dict[str, list[dict]] = {}
    all_pairs: list[dict] = []
    for fname in sorted(by_name):
        occs = by_name[fname]
        if len(occs) < 2:
            continue
        file_pairs: list[dict] = []
        for i in range(len(occs)):
            for j in range(i + 1, len(occs)):
                oa, ob = occs[i], occs[j]
                if oa["run_id"] is not None and ob["run_id"] is not None and oa["run_id"] == ob["run_id"]:
                    continue
                if oa["path"] == ob["path"]:
                    continue
                if oa["item_idx"] == ob["item_idx"]:
                    cos = 1.0
                elif has_vecs:
                    cos = float(vecs[oa["item_idx"]] @ vecs[ob["item_idx"]])
                else:
                    cos = 0.0
                entry = {
                    "name": fname,
                    "a": oa["path"],
                    "b": ob["path"],
                    "run_a": oa["run_id"],
                    "run_b": ob["run_id"],
                    "outcome_a": oa["outcome"],
                    "outcome_b": ob["outcome"],
                    "cosine": round(cos, 4),
                }
                file_pairs.append(entry)
        file_pairs.sort(key=lambda r: (-r["cosine"], r["a"], r["b"]))
        if file_pairs:
            files_out[fname] = file_pairs[:k]
            all_pairs.extend(file_pairs[:k])

    all_pairs.sort(key=lambda r: (-r["cosine"], r["name"], r["a"], r["b"]))
    return {"files": files_out, "pairs": all_pairs}


def make_lineage_fixture(seed, n_items=40):
    """Code items whose vectors are dyadic (every float32 dot product is exact, so a blocked matrix product and a
    single-pair dot agree), with tied 4-decimal cosines, same-run, same-path, same-item and run-less occurrences."""
    rng = np.random.default_rng(seed)
    vecs = (rng.integers(-4, 5, (n_items, 4)) / 8).astype(np.float32)
    runs = [f"r{j}" for j in range(6)] + [None, None]
    items = []
    for i in range(n_items):
        occs = []
        for _ in range(int(rng.integers(1, 4))):    # several occurrences of one item: same item_idx, cosine 1.0
            run = runs[int(rng.integers(0, len(runs)))]
            sub = "ab"[int(rng.integers(0, 2))]     # a small path space, so occurrences share paths
            name = "gate.py" if rng.random() < 0.85 else "util.py"
            occs.append({"path": f"runs/{run or 'loose'}/{sub}/{name}", "run_id": run, "name": name,
                         "suffix": ".py", "outcome": "pass" if i % 2 else "fail"})
        if i % 13 == 5:
            occs.append({"path": f"runs/r0/notes{i}.txt", "run_id": "r0", "name": f"notes{i}.txt",
                         "suffix": ".txt", "outcome": None})
        items.append(desk.TextItem(sha256=f"{i:064x}", paths=[o["path"] for o in occs], name=occs[0]["name"],
                                   suffix=".py", run_id=occs[0]["run_id"], outcome=occs[0]["outcome"],
                                   occurrences=occs))
    return items, vecs


class ManyImagePaths(unittest.TestCase):
    # A tiny block forces many blocks, so every block edge is crossed.
    tiny = mock.patch.object(desk, "BLOCK_BYTES", 4096)

    def test_hash_words_round_trip(self):
        rng = np.random.default_rng(5)
        hs = [rand_hash(rng) for _ in range(20)]
        W = desk._hash_words(hs + [""])
        d = desk._hamming_block(W[:20], W[:20])
        for a in range(20):
            for b in range(20):
                self.assertEqual(int(d[a, b]), desk.hamming(hs[a], hs[b]))
        self.assertFalse(W[20].any())

    def test_popcount_fallback_matches(self):
        rng = np.random.default_rng(6)
        W = desk._hash_words([rand_hash(rng) for _ in range(9)])
        fast = desk._hamming_block(W, W)
        with mock.patch.object(desk, "_HAS_BITWISE_COUNT", False):
            slow = desk._hamming_block(W, W)
        np.testing.assert_array_equal(fast, slow)

    def test_near_pairs_match_the_loop(self):
        cases = [
            make_items(90),
            make_items(90, seed=2, drop_hash={0, 5, 44, 89}),
            make_items(90, seed=3, drop_flip=set(range(0, 90, 4))),   # some rows and columns without a mirror hash
            make_items(90, seed=4, drop_flip=set(range(90))),         # no mirror hashes at all
        ]
        for block_bytes in (4096, 32768):
            for has_bw in (True, False):
                with mock.patch.object(desk, "BLOCK_BYTES", block_bytes), \
                        mock.patch.object(desk, "_HAS_BITWISE_COUNT", has_bw):
                    for items in cases:
                        for near in (0, 10, 26, 44):
                            got = list(desk.near_pairs_within(items, near))
                            self.assertEqual(got, near_reference(items, near))
                            self.assertGreater(len(near_reference(items, 44)), 0)

    def test_dupes_unchanged(self):
        items = make_items(70, seed=7)
        img = unit_vectors(70)
        with self.tiny:
            got = desk.dupes(items, img, near=30)
        ref_near = near_reference(items, 30)
        self.assertEqual(got["near"], ref_near)

    def test_outliers_match(self):
        items = make_items(80, seed=8)
        img = unit_vectors(80)
        with self.tiny:
            got = desk.outliers(items, img, k=5, n=10)
        ref = outliers_reference(items, img, k=5, n=10)
        # A block of rows and the whole matrix can differ in the last float bit (different BLAS paths), as the
        # desk's other blocked similarity code already does; the images and their order must be the same.
        self.assertEqual((got["approved"], got["k"]), (ref["approved"], ref["k"]))
        for key in ("least_typical", "nearest_canon", "furthest_from_canon"):
            self.assertEqual([i for i, _ in got[key]], [i for i, _ in ref[key]], key)
            np.testing.assert_allclose([v for _, v in got[key]], [v for _, v in ref[key]], rtol=1e-6, atol=1e-6)

    def test_look_fit_matches(self):
        items = make_items(80, seed=9)
        with self.tiny:
            got = desk.look_fit(items, k=5)
        ref = look_fit_reference(items, k=5)
        # The matrix-product form of the distance agrees with the difference form to float precision.
        self.assertEqual(list(got), list(ref))
        np.testing.assert_allclose(list(got.values()), list(ref.values()), rtol=1e-5, atol=1e-5)

    def test_block_rows_never_zero(self):
        self.assertEqual(desk._block_rows(10**12, 8), 1)
        self.assertGreater(desk._block_rows(100, 4), 1)

    def _make_described_set(self, seed=10):
        """Build a pool with description groups of 1, 2, and 5 siblings, sibling ties with the own image,
        and byte-identical image vectors (including a non-sibling duplicate of a description's own image)."""
        rng = np.random.default_rng(seed)
        # Group sizes: five groups of 1, three groups of 2, two groups of 5 = 5 + 6 + 10 = 21 described approved items
        # plus 6 unapproved/unrecorded items in the image pool = 27 items total.
        group_specs = [
            ("solo alpha", 1),
            ("solo beta", 1),
            ("solo gamma", 1),
            ("solo delta", 1),
            ("solo epsilon", 1),
            ("pair one", 2),
            ("pair two", 2),
            ("pair three", 2),
            ("quintet north", 5),
            ("quintet south", 5),
        ]
        items = []
        txt_entries = []
        d = 32
        img_list = []
        idx = 0
        for label, count in group_specs:
            base_txt_vec = unit_vectors(1, d=d, seed=seed * 100 + idx + 1)[0]
            for _ in range(count):
                sha = f"{idx:064x}"
                img_v = unit_vectors(1, d=d, seed=seed * 1000 + idx + 1)[0]
                items.append(desk.Item(
                    sha256=sha,
                    paths=[f"outputs/w/{idx}.png"],
                    folders=["w"],
                    dhash=rand_hash(rng),
                    dhash_flip=rand_hash(rng),
                    record={"status": "approved", "id": str(idx), "explanation": label},
                ))
                img_list.append(img_v)
                txt_entries.append((sha, base_txt_vec.copy()))
                idx += 1

        # Add 6 pool-only items (rejected or unrecorded)
        for _ in range(6):
            sha = f"{idx:064x}"
            img_v = unit_vectors(1, d=d, seed=seed * 1000 + idx + 1)[0]
            items.append(desk.Item(
                sha256=sha,
                paths=[f"outputs/w/{idx}.png"],
                folders=["w"],
                dhash=rand_hash(rng),
                dhash_flip=rand_hash(rng),
                record={"status": "rejected", "id": str(idx), "explanation": None},
            ))
            img_list.append(img_v)
            idx += 1

        img = np.vstack(img_list).astype(np.float32)
        # Force byte-identical image vectors:
        # 1) Non-sibling pool image (index 21) is byte-identical to solo alpha's own image (index 0)
        img[21] = img[0].copy()
        # 2) In "pair one" (indices 5, 6), make sibling 6 byte-identical to own image 5 (sibling tie with own image)
        img[6] = img[5].copy()
        # 3) In "quintet north" (indices 11..15), make siblings 12 and 14 byte-identical to 11, and pool image 22
        #    also byte-identical to 11 (both sibling and non-sibling ties with own image)
        img[12] = img[11].copy()
        img[14] = img[11].copy()
        img[22] = img[11].copy()
        # 4) In "quintet south" (indices 16..20), make 17 and 18 byte-identical to each other
        img[18] = img[17].copy()

        # Shuffle txt insertion order deterministically to test that captions() preserves txt iteration order on ties
        order = rng.permutation(len(txt_entries))
        txt = {txt_entries[int(i)][0]: txt_entries[int(i)][1] for i in order}
        return items, img, txt

    def test_captions_and_swap_check_equivalence_and_byte_identical_images(self):
        for seed in (10, 11, 12, 13, 14):
            items, img, txt = self._make_described_set(seed=seed)
            n_cols = len(img)
            cell_bytes = img.itemsize
            # Test block sizes where _block_rows == 1 and where _block_rows == 4 (>= 3, crossing block edges)
            for block_bytes in (1, n_cols * cell_bytes * 4):
                with mock.patch.object(desk, "BLOCK_BYTES", block_bytes):
                    rows_per_block = desk._block_rows(n_cols, cell_bytes)
                    if block_bytes == 1:
                        self.assertEqual(rows_per_block, 1)
                    else:
                        self.assertGreaterEqual(rows_per_block, 3)

                    got_cap = desk.captions(items, img, txt, n=25)
                    ref_cap = captions_reference(items, img, txt, n=25)
                    for k in ("described", "pool", "recall_at_1", "recall_at_5",
                              "median_rank", "group_recall_at_1", "group_median_rank"):
                        self.assertEqual(got_cap[k], ref_cap[k], f"captions[{k}] mismatch at seed={seed}, block_bytes={block_bytes}")
                    self.assertEqual(len(got_cap["worst"]), len(ref_cap["worst"]))
                    for g_row, r_row in zip(got_cap["worst"], ref_cap["worst"], strict=True):
                        self.assertEqual(g_row["item"], r_row["item"])
                        self.assertEqual(g_row["rank"], r_row["rank"])
                        self.assertEqual(g_row["group_rank"], r_row["group_rank"])
                        self.assertEqual(g_row["siblings"], r_row["siblings"])
                        # Every rank and group_rank must be an integer or a half-integer
                        self.assertEqual((2.0 * g_row["rank"]) % 1.0, 0.0)
                        self.assertEqual((2.0 * g_row["group_rank"]) % 1.0, 0.0)
                        np.testing.assert_allclose(g_row["cosine"], r_row["cosine"], rtol=1e-6, atol=1e-6)

                    got_sw = desk.swap_check(items, img, txt, seed=seed)
                    ref_sw = swap_check_reference(items, img, txt, seed=seed)
                    self.assertEqual(got_sw, ref_sw, f"swap_check mismatch at seed={seed}, block_bytes={block_bytes}")

        # Empty T edge case on _rank_descriptions
        own_r, own_c, sec_r, grp_r = desk._rank_descriptions(
            np.zeros((4, 8), dtype=np.float32),
            np.zeros((0, 8), dtype=np.float32),
            [],
            second_idx=[],
            sibling_groups=[],
        )
        self.assertEqual(len(own_r), 0)
        self.assertEqual(len(own_c), 0)
        self.assertEqual(len(sec_r), 0)
        self.assertEqual(len(grp_r), 0)

    def test_auc_ties_match_pairwise_formula_exactly(self):
        def auc_pairwise(t: np.ndarray, w: np.ndarray) -> float:
            return float(((w[:, None] > t[None, :]).sum() + 0.5 * (w[:, None] == t[None, :]).sum()) / (len(w) * len(t)))

        def auc_sorted(t: np.ndarray, w: np.ndarray) -> float:
            t_s = np.sort(t)
            left = np.searchsorted(t_s, w, side="left")
            right = np.searchsorted(t_s, w, side="right")
            return float((left.sum() + 0.5 * (right - left).sum()) / (len(w) * len(t)))

        handmade_cases = [
            # All equal across both arrays
            (np.array([2.0, 2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0, 2.0])),
            # Equal plateaus within each array and overlapping across arrays (including half-integer ranks)
            (np.array([1.0, 1.5, 1.5, 2.0, 3.5, 3.5, 3.5, 5.0]),
             np.array([1.5, 1.5, 2.0, 2.0, 2.5, 3.5, 5.0, 5.0])),
            # Reversed order with duplicates
            (np.array([5.0, 3.5, 3.5, 1.0, 1.0, 2.5]),
             np.array([3.5, 1.0, 5.0, 5.0, 2.5, 2.5])),
            # Disjoint ties
            (np.array([1.0, 1.0, 1.0]), np.array([4.0, 4.0, 4.0])),
            (np.array([4.0, 4.0, 4.0]), np.array([1.0, 1.0, 1.0])),
        ]
        for t, w in handmade_cases:
            self.assertEqual(auc_sorted(t, w), auc_pairwise(t, w))

        rng = np.random.default_rng(42)
        for _ in range(50):
            m = int(rng.integers(3, 60))
            # Half-integer ranks with frequent ties
            t = rng.integers(2, 25, size=m).astype(np.float64) * 0.5
            w = rng.integers(2, 25, size=m).astype(np.float64) * 0.5
            self.assertEqual(auc_sorted(t, w), auc_pairwise(t, w))

    def test_swap_check_memory_bound_at_3000_descriptions(self):
        m = 3000
        d = 8
        rng = np.random.default_rng(99)
        img = rng.standard_normal((m, d)).astype(np.float32)
        img /= np.linalg.norm(img, axis=1, keepdims=True)
        items = [
            desk.Item(
                sha256=f"{i:064x}",
                paths=[f"outputs/w/{i}.png"],
                folders=["w"],
                dhash="",
                record={"status": "approved", "id": str(i), "explanation": f"unique caption {i}"},
            )
            for i in range(m)
        ]
        txt = {
            f"{i:064x}": img[(i + 1) % m].copy()
            for i in range(m)
        }
        # One m x m boolean array is 9,000,000 bytes (~8.58 MiB); two are ~17.16 MiB.
        # With BLOCK_BYTES = 65536, peak traced memory inside swap_check must stay well below one m x m array (< 4 MiB).
        one_m_by_m_bytes = m * m
        bound_bytes = 4 * 1024 * 1024
        self.assertLess(bound_bytes, one_m_by_m_bytes)

        with mock.patch.object(desk, "BLOCK_BYTES", 65536):
            tracemalloc.start()
            try:
                res = desk.swap_check(items, img, txt, seed=0)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        self.assertIsNotNone(res)
        self.assertEqual(res["pairs"], m)
        self.assertLess(peak, bound_bytes, f"swap_check peak memory {peak} bytes exceeded bound {bound_bytes} bytes")

    def test_conv1d_gauss2d_palette_and_parallel_scan_match_020_verbatim(self):
        def conv1d_020(a: np.ndarray, k: np.ndarray) -> np.ndarray:
            r = len(k) // 2
            p = np.pad(a, ((r, r), (0, 0)), mode="reflect")
            out = np.zeros_like(a, dtype=np.float32)
            for i, w in enumerate(k):
                out += w * p[i:i + a.shape[0]]
            return out

        def gauss2d_020(a: np.ndarray, sigma: float) -> np.ndarray:
            r = max(1, int(round(3.0 * sigma)))
            x = np.arange(-r, r + 1, dtype=np.float32)
            k = np.exp(-0.5 * (x / sigma) ** 2)
            k /= k.sum()
            return conv1d_020(conv1d_020(a, k).T, k).T

        def texture_stats_020(im) -> list[float]:
            rgb_im = im.convert("RGB")
            w, h = rgb_im.size
            if w < 8 or h < 8:
                return [0.0] * 6
            if max(w, h) > 1344:
                scale = 1344.0 / max(w, h)
                rgb_im = rgb_im.resize(
                    (max(8, int(round(w * scale))), max(8, int(round(h * scale)))),
                    Image.Resampling.LANCZOS,
                )
            a = np.asarray(rgb_im, dtype=np.float32)
            lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
            lum_std = float(lum.std())
            if lum_std < 1e-6:
                return [0.0] * 6

            hp = lum - gauss2d_020(lum, 1.5)
            struct = gauss2d_020(lum, 6.0)
            sgx, sgy = desk._sobel(struct)
            sgrad = np.hypot(sgx, sgy)
            flat = sgrad <= np.percentile(sgrad, 40)

            ex, ey = desk._sobel(lum)
            egrad = np.hypot(ex, ey) / 4.0
            blur = gauss2d_020(lum, 2.0)
            bx, by = desk._sobel(blur)
            bgrad = np.hypot(bx, by) / 4.0
            strong = egrad >= np.percentile(egrad, 98)
            raw_hard = float(egrad[strong].mean() / max(float(bgrad[strong].mean()), 1e-6)) if strong.any() else 0.0

            hx, hy = desk._sobel(hp)
            jxx = gauss2d_020(hx * hx, 4.0)
            jyy = gauss2d_020(hy * hy, 4.0)
            jxy = gauss2d_020(hx * hy, 4.0)
            disc = np.sqrt(((jxx - jyy) / 2.0) ** 2 + jxy ** 2)
            tr = (jxx + jyy) / 2.0
            coh = 2.0 * disc / np.maximum(tr * 2.0, 1e-6)

            win = np.hanning(lum.shape[0])[:, None] * np.hanning(lum.shape[1])[None, :]
            spec = np.fft.fftshift(np.abs(np.fft.fft2((lum - float(lum.mean())) * win)) ** 2)
            cy, cx = (np.array(spec.shape) - 1) / 2.0
            yy, xx = np.ogrid[:spec.shape[0], :spec.shape[1]]
            rad = np.hypot((yy - cy) / max(spec.shape[0] / 2.0, 1.0), (xx - cx) / max(spec.shape[1] / 2.0, 1.0))
            tot = max(float(spec.sum()), 1e-9)

            hf_ratio = min(1.0, float(hp.std()) / max(lum_std, 1e-6))
            flat_hf = min(1.0, float(hp[flat].std()) / 32.0) if flat.any() else 0.0
            edge_dens = float((egrad > 12.0).mean())
            edge_hard = float(np.tanh(raw_hard / 3.0))
            flat_coh = float(coh[flat].mean()) if flat.any() else 0.0
            band_hi = float(spec[rad > 0.30].sum() / tot)
            return [hf_ratio, flat_hf, edge_dens, edge_hard, flat_coh, band_hi]

        def palette_020(im) -> list[float]:
            hsv = np.asarray(im.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32) / 255.0
            h, sat, val = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
            hue = np.histogram(h, bins=12, range=(0, 1), weights=sat)[0]
            hue = hue / max(float(hue.sum()), 1e-6)
            tone = np.histogram(val, bins=8, range=(0, 1))[0] / val.size
            tex = texture_stats_020(im)
            vec = np.concatenate([hue, tone, [sat.mean(), sat.std(), val.mean(), val.std()], tex])
            return [round(float(x), 5) for x in vec]

        rng = np.random.default_rng(2026)
        for shape in ((240, 1024), (300, 200), (8, 8)):
            arr = rng.random(shape, dtype=np.float32) * 255.0
            for sigma in (1.5, 2.0, 4.0, 6.0):
                self.assertTrue(
                    np.array_equal(desk._gauss2d(arr, sigma), gauss2d_020(arr, sigma)),
                    f"_gauss2d mismatch at shape={shape}, sigma={sigma}",
                )

        for w, h in ((240, 1024), (1024, 1024), (1600, 1200), (8, 8)):
            px = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            im = Image.fromarray(px, mode="RGB")
            self.assertEqual(desk._palette(im), palette_020(im), f"_palette mismatch at size=({w}, {h})")

        # Verify scan(threads=1) and scan(threads=4) produce identical items, problems, and scan.jsonl rows
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "proj")
            wdir = os.path.join(proj, "outputs", "w1")
            os.makedirs(wdir)
            for i in range(10):
                px = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(px, mode="RGB").save(os.path.join(wdir, f"img_{i:02d}.png"))
            with open(os.path.join(wdir, "img_05_corrupt.png"), "wb") as f:
                f.write(b"not a valid png")

            out1 = os.path.join(tmp, "out1")
            out4 = os.path.join(tmp, "out4")
            items1, stats1 = desk.scan(proj, strict=False, out=out1, threads=1)
            items4, stats4 = desk.scan(proj, strict=False, out=out4, threads=4)
            self.assertEqual(items1, items4)
            self.assertEqual(stats1["problems"], stats4["problems"])
            self.assertEqual(len(stats1["problems"]), 1)
            self.assertEqual(stats1["look_computed_now"], 10)
            self.assertEqual(stats4["look_computed_now"], 10)

            with open(os.path.join(out1, "scan.jsonl"), encoding="utf-8") as f1, \
                    open(os.path.join(out4, "scan.jsonl"), encoding="utf-8") as f4:
                rows1 = [{k: v for k, v in json.loads(line).items() if k != "look_seconds"} for line in f1]
                rows4 = [{k: v for k, v in json.loads(line).items() if k != "look_seconds"} for line in f4]
            self.assertEqual(rows1, rows4)

            with self.assertRaises((OSError, ValueError, SyntaxError)):
                desk.scan(proj, strict=True, threads=4)

            # Worker initializer sets single-thread BLAS env vars
            with mock.patch.dict(os.environ):
                desk._init_scan_worker()
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
                self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
                self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")

    def test_description_gate_prefix_filter_matches_bruteforce_on_600_records(self):
        rng = np.random.default_rng(777)
        vocab = [f"word{i:04d}" for i in range(400)]
        recs = []

        def add_rec(rid: str, text: str):
            recs.append({
                "id": rid,
                "asset_path": f"outputs/w/{rid}.png",
                "judgment": {"status": "approved", "explanation": text},
                "provenance": {"prompt": "unrelated prompt words here"},
            })

        # 1) 440 random descriptions with varying lengths (8 to 28 content words)
        for i in range(440):
            k = int(rng.integers(8, 29))
            chosen = rng.choice(vocab, size=k, replace=False)
            add_rec(f"u_{i:03d}", " ".join(chosen))

        # 2) 30 near-miss pairs around 0.80 - 0.8421 (e.g. 16 shared / 19 union = 0.8421, 13/16 = 0.8125, 17/21 = 0.8095)
        for i in range(30):
            base = [f"miss_{i}_{w}" for w in range(16)]
            add_rec(f"miss_a_{i:02d}", " ".join([*base, f"ma1_{i}", f"ma2_{i}"]))
            add_rec(f"miss_b_{i:02d}", " ".join([*base, f"mb1_{i}"]))

        # 3) 20 boundary pairs at exact 0.85 (17 shared / 20 union = 0.85)
        for i in range(20):
            base = [f"bnd_{i}_{w}" for w in range(17)]
            add_rec(f"bnd_a_{i:02d}", " ".join([*base, f"ba1_{i}", f"ba2_{i}"]))
            add_rec(f"bnd_b_{i:02d}", " ".join([*base, f"bb1_{i}"]))

        # 4) 15 high-similarity near-duplicate pairs (0.90 to 0.96) + triples where one record has multiple near partners
        for i in range(15):
            base = [f"hi_{i}_{w}" for w in range(20)]
            add_rec(f"hi_a_{i:02d}", " ".join([*base, f"ha_{i}"]))                 # shares 20/21 = 0.9524 with hi_b, 19/21 = 0.9048 with hi_c
            add_rec(f"hi_b_{i:02d}", " ".join(base))                                # shares 20/21 with hi_a, 19/20 = 0.95 with hi_c
            add_rec(f"hi_c_{i:02d}", " ".join([*base[:19], f"hc_{i}"]))

        # 5) Two 5-record exact-duplicate clusters (including punctuation variations)
        for c in range(2):
            for m in range(5):
                sep = "-" if m % 2 else " "
                add_rec(
                    f"exact_{c}_{m}",
                    f"lantern{sep}courtyard stone pillar bronze archway mist harbour {c}",
                )

        # 6) 5 stopword-only records
        for s in range(5):
            add_rec(f"stop_{s}", "the of and to in with by on from")

        self.assertEqual(len(recs), 600)
        audit = description_gate.audit_records(recs)

        # Brute-force O(n^2) reference over the same target_list and exact_keys
        ids_ordered = [r["id"] for r in audit["records"]]
        cword_sets = {r["id"]: description_gate.content_word_set(r["text"]) for r in audit["records"]}
        exact_keys: dict[str, set[tuple[str, str]]] = {}
        for g in audit["exact_groups"]:
            for rid in g["ids"]:
                exact_keys.setdefault(rid, set()).add((g["source"], g["normalized"]))

        ref_near_pairs: list[tuple[str, str, float]] = []
        ref_max_jac: dict[str, float | None] = {rid: None for rid in ids_ordered}
        eps = description_gate._EPS
        thr = description_gate.MAX_CONTENT_JACCARD
        for i in range(len(ids_ordered)):
            ia = ids_ordered[i]
            ca = cword_sets[ia]
            if not ca:
                continue
            keys_a = exact_keys.get(ia)
            for j in range(i + 1, len(ids_ordered)):
                ib = ids_ordered[j]
                if keys_a and keys_a & exact_keys.get(ib, set()):
                    continue
                cb = cword_sets[ib]
                if not cb:
                    continue
                n_inter = len(ca & cb)
                sim = n_inter / (len(ca) + len(cb) - n_inter)
                if sim >= thr - eps:
                    r_sim = round(sim, 6)
                    if ref_max_jac[ia] is None or r_sim > ref_max_jac[ia]:
                        ref_max_jac[ia] = r_sim
                    if ref_max_jac[ib] is None or r_sim > ref_max_jac[ib]:
                        ref_max_jac[ib] = r_sim
                    ref_near_pairs.append((ia, ib, round(sim, 4)))

        ref_near_groups = description_gate._connected_components(
            ids_ordered, [(a, b) for a, b, _ in ref_near_pairs]
        )
        self.assertEqual(audit["near_pairs"], ref_near_pairs)
        self.assertEqual(audit["near_groups"], ref_near_groups)
        for r in audit["records"]:
            self.assertEqual(r["max_content_jaccard"], ref_max_jac[r["id"]], f"mismatch on {r['id']}")

    def test_blocked_pairs_top1_lineage_and_coverage_scale_paths(self):
        vecs = unit_vectors(35, d=8, seed=31)
        ref_pairs = list(desk.pairs_at_least(vecs, 0.25, chunk=100))
        ref_top1 = desk.top1_percentiles(vecs, chunk=100)
        with mock.patch.object(desk, "BLOCK_BYTES", 64):
            got_pairs = list(desk.pairs_at_least(vecs, 0.25))
            self.assertEqual([(a, b) for a, b, _ in got_pairs], [(a, b) for a, b, _ in ref_pairs])
            np.testing.assert_allclose([s for _, _, s in got_pairs], [s for _, _, s in ref_pairs], rtol=1e-6, atol=1e-6)
            self.assertEqual(desk.top1_percentiles(vecs), ref_top1)

        # Lineage blocked matrix product with tiny BLOCK_BYTES
        t_items = [
            desk.TextItem(
                sha256=f"{i:064x}",
                paths=[f"runs/r{i}/files/gate.py"],
                name="gate.py",
                suffix=".py",
                run_id=f"r{i}",
                outcome="pass" if i % 2 == 0 else "fail",
                occurrences=[{
                    "path": f"runs/r{i}/files/gate.py",
                    "run_id": f"r{i}",
                    "name": "gate.py",
                    "suffix": ".py",
                    "outcome": "pass" if i % 2 == 0 else "fail",
                }],
            )
            for i in range(12)
        ]
        # Add a second occurrence of item 0 in another run (same item_idx -> cosine 1.0) and in same run (skipped)
        t_items[0].occurrences.append({
            "path": "runs/r0/files/sub/gate.py",
            "run_id": "r0",
            "name": "gate.py",
            "suffix": ".py",
            "outcome": "pass",
        })
        t_items[0].occurrences.append({
            "path": "runs/r99/files/gate.py",
            "run_id": "r99",
            "name": "gate.py",
            "suffix": ".py",
            "outcome": "pass",
        })
        t_vecs = unit_vectors(12, d=8, seed=32)
        lin_full = desk.lineage(t_items, t_vecs, k=5)
        with mock.patch.object(desk, "BLOCK_BYTES", 16):
            lin_blocked = desk.lineage(t_items, t_vecs, k=5)
            lin_no_vecs = desk.lineage(t_items, np.zeros((12, 0), dtype=np.float32), k=5)
        self.assertEqual(lin_blocked, lin_full)
        self.assertEqual(lin_blocked["files"]["gate.py"][0]["cosine"], 1.0)
        self.assertEqual(lin_no_vecs["files"]["gate.py"][0]["cosine"], 1.0)
        self.assertEqual(lin_no_vecs["files"]["gate.py"][2]["cosine"], 0.0)

        # Coverage with no approved items vs approved items
        items_unapproved = [
            desk.Item(f"{i:064x}", [f"outputs/w/{i}.png"], ["w"], record={"status": "rejected", "id": str(i)})
            for i in range(4)
        ]
        qv = unit_vectors(2, d=8, seed=33)
        cov_empty = desk.coverage(items_unapproved, vecs[:4], ["s1", "s2"], qv)
        self.assertEqual([r["top"] for r in cov_empty], [[], []])

    def test_lineage_top_k_matches_020_on_ties_and_blocks_and_scales(self):
        no_vecs = np.zeros((40, 0), dtype=np.float32)
        tied_names = 0
        none_pairs = 0
        for seed in range(6):
            items, vecs = make_lineage_fixture(seed)
            m_gate = sum(o["name"] == "gate.py" for it in items for o in it.occurrences)
            # One row per few: BLOCK_BYTES this small forces several blocks over the gate.py occurrences.
            small = m_gate * 8 * 4
            for v in (vecs, no_vecs):
                for k in (1, 5, 10**6):
                    ref = lineage_020(items, v, k=k)
                    self.assertGreater(len(ref["pairs"]), 0)
                    for block_bytes in (desk.BLOCK_BYTES, small):
                        with mock.patch.object(desk, "BLOCK_BYTES", block_bytes):
                            if block_bytes == small:
                                self.assertLess(desk._block_rows(m_gate, 8), m_gate // 3)
                            got = desk.lineage(items, v, k=k)
                        self.assertEqual(got, ref, f"seed={seed} k={k} vecs={v.shape} block_bytes={block_bytes}")
                    if k == 10**6:
                        for pairs in ref["files"].values():
                            cos = [p["cosine"] for p in pairs]
                            tied_names += len(cos) != len(set(cos))
                            none_pairs += sum(p["run_a"] is None and p["run_b"] is None for p in pairs)
        self.assertGreater(tied_names, 0)   # ties are broken by (a, b), not by the order pairs were met
        self.assertGreater(none_pairs, 0)   # two None run ids are not a same-run pair

        # Scale: one script byte-identical across 5,000 runs, so every pair is cos = 1.0 and every pair ties.
        m = 5000
        occs = [{"path": f"runs/run{j:04d}/gate.py", "run_id": f"run{j:04d}", "name": "gate.py", "suffix": ".py",
                 "outcome": "pass"} for j in range(m)]
        big = [desk.TextItem(sha256="0" * 64, paths=[o["path"] for o in occs], name="gate.py", suffix=".py",
                             run_id="run0000", outcome="pass", occurrences=occs)]
        t0 = time.perf_counter()
        res = desk.lineage(big, np.ones((1, 4), dtype=np.float32) / 2, k=5)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"lineage took {elapsed:.2f} s on {m} identical occurrences")
        got = [(p["a"], p["b"]) for p in res["files"]["gate.py"]]
        self.assertEqual(got, [("runs/run0000/gate.py", f"runs/run{j:04d}/gate.py") for j in range(1, 6)])
        self.assertTrue(all(p["cosine"] == 1.0 for p in res["pairs"]))

    def test_rank_descriptions_ties_identical_across_block_sizes(self):
        n_img, m, d = 103, 17, 768
        img = unit_vectors(n_img, d=d, seed=41)
        img[50] = img[0]
        img[102] = img[0]   # three byte-identical images
        T = unit_vectors(m, d=d, seed=42)
        rng = np.random.default_rng(43)
        own_idx = rng.integers(0, n_img, size=m)
        own_idx[0], own_idx[5], own_idx[11] = 0, 50, 102
        second_idx = rng.integers(0, n_img, size=m)
        second_idx[3] = 50
        sibling_groups = [[int(o)] for o in own_idx]
        sibling_groups[0] = [0, 7]
        sibling_groups[5] = [50, 102, 9]

        results = []
        for block_bytes in (1, n_img * img.itemsize * 4):
            with mock.patch.object(desk, "BLOCK_BYTES", block_bytes):
                rows = desk._block_rows(n_img, img.itemsize)
                if block_bytes == 1:
                    self.assertEqual(rows, 1)
                else:
                    self.assertTrue(1 < rows < m)
                own_r, _, sec_r, grp_r = desk._rank_descriptions(
                    img, T, own_idx, second_idx=second_idx, sibling_groups=sibling_groups)
            results.append((own_r, sec_r, grp_r))
        # Ranks only: own cosines may differ in the last bit between block sizes by design.
        for a, b in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(a, b)

        # An identical-vector image gets half-tie credit: 1 + above + 0.5 * (ties - 1), with ties = 3.
        own_r = results[0][0]
        img64 = img.astype(np.float64)
        for k in (0, 5, 11):
            sims = img64 @ T[k].astype(np.float64)
            above = int((sims > sims[own_idx[k]]).sum())
            self.assertEqual(own_r[k], 1 + above + 0.5 * (3 - 1))


if __name__ == "__main__":
    unittest.main()
