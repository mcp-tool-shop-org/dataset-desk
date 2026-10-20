"""The many-image paths give the same results as the per-pair code they replaced, across block boundaries.
Run from the repository root: python -W error -m unittest discover -s tests"""

import tracemalloc
import unittest
from unittest import mock

import numpy as np

from dataset_desk import desk


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
        with self.tiny:
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


if __name__ == "__main__":
    unittest.main()
