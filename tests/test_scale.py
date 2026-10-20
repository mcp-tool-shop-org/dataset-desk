"""The many-image paths give the same results as the per-pair code they replaced, across block boundaries.
Run from the repository root: python -W error -m unittest discover -s tests"""

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


if __name__ == "__main__":
    unittest.main()
