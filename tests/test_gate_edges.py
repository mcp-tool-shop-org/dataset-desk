"""Edge cases of the model-free description gate: helpers, field extraction, query manifests, token counters,
drafting, sibling selection and the review files. Offline; the SigLIP2 tokenizer is a stand-in."""

import json
import os
import tempfile
import unittest

import fakes
from dataset_desk import description_gate as g

GOOD = "a weathered bronze lantern hanging beside carved stone pillars in a quiet courtyard"
OTHER = "two grey herons wading through shallow reeds under pale morning mist near a river"


def rec(rid, text, *, status="approved", prompt="", subject="", asset=None, author_kind=None):
    r = {"id": rid, "asset_path": asset if asset is not None else f"outputs/{rid}.png",
         "judgment": {"status": status, "explanation": text}, "provenance": {"prompt": prompt}}
    if subject:
        r["canon"] = {"subject": subject}
    if author_kind:
        r["provenance"]["author_kind"] = author_kind
    return r


class Helpers(unittest.TestCase):
    def test_word_helpers(self):
        self.assertEqual(g.normalize_text("A b-C, d!"), "abcd")
        self.assertEqual(g.word_set("a a B"), {"a", "b"})
        self.assertEqual(g.content_word_set("The red cat and a dog"), {"red", "cat", "dog"})

    def test_content_jaccard_strings_sets_and_empty(self):
        self.assertEqual(g.content_jaccard("red cat dog", "red cat dog"), 1.0)
        self.assertAlmostEqual(g.content_jaccard("red cat", "red dog"), 1 / 3)
        self.assertEqual(g.content_jaccard({"x", "y"}, {"y"}), 0.5)
        self.assertEqual(g.content_jaccard("the of and", "red cat"), 0.0)
        self.assertEqual(g.content_jaccard(set(), {"a"}), 0.0)
        self.assertIs(g.jaccard, g.content_jaccard)

    def test_word_share_and_runs_with_empty_sides(self):
        self.assertEqual(g.word_share([], ["a"]), 0.0)
        self.assertEqual(g.word_share(["a"], []), 0.0)
        self.assertEqual(g.word_share(["a", "a", "b"], ["a"]), 2 / 3)
        self.assertEqual(g.max_contiguous_run([], ["a"]), 0)
        self.assertEqual(g.max_contiguous_run(["a"], []), 0)
        self.assertEqual(g.max_contiguous_run(list("xabcy"), list("zabcq")), 3)

    def test_wraps_canon_subject_edges(self):
        self.assertFalse(g.wraps_canon_subject("anything at all", "too short"))   # under 4 subject words
        self.assertFalse(g.wraps_canon_subject("", "four words are here"))        # empty description
        self.assertFalse(g.wraps_canon_subject("not the same words at all", "four words are here"))
        self.assertTrue(g.wraps_canon_subject("so four words are here, really", "Four words are here"))

    def test_prompt_echo_equal_and_inside(self):
        eq = g.prompt_echo_stats("Red cat sits", "red cat sits")
        self.assertTrue(eq["equals_prompt"])
        self.assertFalse(eq["inside_prompt"])
        inside = g.prompt_echo_stats("red cat", "a red cat sits")
        self.assertTrue(inside["inside_prompt"])
        self.assertFalse(inside["equals_prompt"])


class SigLIPTokenizer(unittest.TestCase):
    def fake_tf(self):
        tf = fakes.make_transformers()
        tf.seen = []

        class Tok:
            def __call__(self, s, padding, truncation):
                tf.seen.append((s, padding, truncation))
                return {"input_ids": list(range(len(s.split()) + 2))}  # words plus two special tokens

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(repo, **kw):
                tf.seen.append(("load", repo, kw))
                return Tok()

        tf.AutoTokenizer = AutoTokenizer
        return tf

    def test_loader_is_pinned_and_local_only(self):
        from dataset_desk import backends

        tf = self.fake_tf()
        with fakes.fake_modules(transformers=tf):
            g.load_siglip2_tokenizer()
        self.assertEqual(tf.seen[0], ("load", backends.SIGLIP2_REPO,
                                      {"revision": backends.SIGLIP2_REVISION, "local_files_only": True}))

    def test_count_lowercases_and_loads_lazily(self):
        tf = self.fake_tf()
        with fakes.fake_modules(transformers=tf):
            self.assertEqual(g.siglip2_token_count("Red CAT"), 4)
            self.assertEqual(g.siglip2_token_count(None), 2)  # empty text: just the two special tokens
        calls = [c for c in tf.seen if c[0] != "load"]
        self.assertEqual(calls[0], ("red cat", False, False))

    def test_counts_list_empty_does_not_load_and_nonempty_loads_once(self):
        tf = self.fake_tf()
        with fakes.fake_modules(transformers=tf):
            self.assertEqual(g.siglip2_token_counts([]), [])
            self.assertEqual(tf.seen, [])
            self.assertEqual(g.siglip2_token_counts(["a b", "c"]), [4, 3])
        self.assertEqual(len([c for c in tf.seen if c[0] == "load"]), 1)

    def test_explicit_tokenizer_is_used_without_loading(self):
        tok = lambda s, padding, truncation: {"input_ids": [0] * 7}  # noqa: E731
        self.assertEqual(g.siglip2_token_count("x", tokenizer=tok), 7)
        self.assertEqual(g.siglip2_token_counts(["x", "y"], tokenizer=tok), [7, 7])


class ExtractFields(unittest.TestCase):
    def test_raw_record_with_canon_fallback_when_unapproved_or_no_explanation(self):
        e = g._extract_record_fields({"id": "a", "judgment": {"status": "rejected", "explanation": "text here"},
                                      "canon": {"subject": "the subject"}, "provenance": {"author_kind": "human"}})
        self.assertEqual((e["text"], e["source"]), ("the subject", "canon.subject"))
        self.assertEqual(e["existing_author_kind"], "human")
        e = g._extract_record_fields({"id": "b", "judgment": {"status": "approved"}})
        self.assertEqual((e["text"], e["source"]), ("", "none"))

    def test_raw_author_kind_may_live_on_judgment_and_paths_are_normalised(self):
        e = g._extract_record_fields({"id": "a", "asset_path": "x\\y.png",
                                      "judgment": {"status": "approved", "explanation": "e", "author_kind": "human"}})
        self.assertEqual((e["asset_path"], e["existing_author_kind"]), ("x/y.png", "human"))

    def test_flat_desk_item_forms(self):
        e = g._extract_record_fields({"id": "a", "status": "approved", "judgment_note": "note text",
                                      "canon_subject": "s", "prompt": "p", "author_kind": "human"})
        self.assertEqual((e["text"], e["source"], e["prompt"]), ("note text", "judgment.explanation", "p"))
        e = g._extract_record_fields({"id": "b", "status": "approved", "judgment_explanation": "ex"})
        self.assertEqual(e["judgment_explanation"], "ex")
        e = g._extract_record_fields({"id": "c", "status": "approved", "explanation": "plain explanation"})
        self.assertEqual(e["text"], "plain explanation")
        e = g._extract_record_fields({"id": "d", "status": "approved", "canon_subject": "only subject"})
        self.assertEqual((e["text"], e["source"]), ("only subject", "canon.subject"))


class QueriesAndCounters(unittest.TestCase):
    def setUp(self):
        self.records = [rec("r1", GOOD, subject="lantern pillars courtyard scene"), rec("r2", OTHER)]

    def by_id(self, audit):
        return {r["id"]: r for r in audit["records"]}

    def test_query_manifest_sources_and_flags(self):
        q = {"r1": GOOD,                                    # same as current explanation
             "r2": "lantern pillars courtyard scene",       # a different text; not r2's own
             "ghost": OTHER,                                # not in the records at all
             "r3": ""}                                      # empty query
        a = g.audit_records(self.records + [rec("r3", GOOD + " extra")], queries=q)
        rows = self.by_id(a)
        self.assertEqual(rows["r1"]["source"], "judgment.explanation")
        self.assertEqual(rows["r2"]["source"], "custom")
        self.assertIn("not_current_description", rows["r2"]["flags"])
        self.assertIn("query text does not match record's current judgment.explanation", rows["r2"]["reasons"])
        self.assertEqual(rows["ghost"]["status"], "missing")
        self.assertIn("not_approved", rows["ghost"]["flags"])
        self.assertIn("status is 'missing', not 'approved'", rows["ghost"]["reasons"])
        self.assertEqual(rows["r3"]["source"], "none")
        self.assertIn("empty_description", rows["r3"]["flags"])
        self.assertIn("no approved description text", rows["r3"]["reasons"])
        self.assertEqual(a["checked"], 4)

    def test_query_equal_to_canon_subject_is_canon_source_with_reason(self):
        a = g.audit_records(self.records, queries={"r1": "lantern pillars courtyard scene"})
        row = a["records"][0]
        self.assertEqual(row["source"], "canon.subject")
        self.assertIn("not_current_description", row["flags"])
        self.assertIn("text comes from canon.subject, not judgment.explanation", row["reasons"])

    def test_equals_prompt_reason(self):
        a = g.audit_records([rec("p", GOOD, prompt=GOOD)])
        row = a["records"][0]
        self.assertIn("prompt_echo", row["flags"])
        self.assertIn("equals generation prompt", row["reasons"])
        self.assertTrue(row["equals_prompt"])

    def test_callable_counter_marks_checked_and_flags_too_long(self):
        a = g.audit_records(self.records, token_counter=lambda texts: [10, 65])
        rows = self.by_id(a)
        self.assertEqual((rows["r1"]["gate_status"], rows["r1"]["siglip2_tokens"]), ("passed", 10))
        self.assertEqual(rows["r2"]["gate_status"], "failed")
        self.assertIn("too_long_siglip2", rows["r2"]["flags"])
        self.assertIn("SigLIP2 token count 65 > 64", rows["r2"]["reasons"])
        self.assertEqual((a["passed_count"], a["flagged_count"], a["unchecked_count"]), (1, 1, 0))

    def test_callable_counter_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            g.audit_records(self.records, token_counter=lambda texts: [1])

    def test_dict_counter_by_id_by_text_and_missing(self):
        a = g.audit_records(self.records + [rec("r3", "a rusted iron gate swinging open onto an overgrown walled orchard garden")],
                            token_counter={"r1": 12, OTHER: 70})
        rows = self.by_id(a)
        self.assertEqual(rows["r1"]["siglip2_tokens"], 12)
        self.assertEqual(rows["r2"]["siglip2_tokens"], 70)
        self.assertIn("too_long_siglip2", rows["r2"]["flags"])
        self.assertIsNone(rows["r3"]["siglip2_tokens"])
        self.assertEqual(rows["r3"]["token_status"], "unchecked")
        self.assertEqual(rows["r3"]["gate_status"], "unchecked")

    def test_no_counter_leaves_clean_records_unchecked_not_passed(self):
        a = g.audit_records(self.records)
        self.assertTrue(a["passed_model_free_gate"])
        self.assertFalse(a["passed_gate"])
        self.assertEqual(a["unchecked_count"], 2)

    def test_empty_records_never_pass(self):
        a = g.audit_records([])
        self.assertFalse(a["passed_gate"])
        self.assertFalse(a["passed_model_free_gate"])
        self.assertEqual((a["checked"], a["max_exact_group"], a["median_prompt_word_share"]), (0, 0, 0.0))

    def test_stopword_only_descriptions_skip_jaccard_comparison(self):
        a = g.audit_records([rec("s1", "the of and"), rec("s2", "the of and to"), rec("ok", GOOD)])
        rows = self.by_id(a)
        self.assertEqual(rows["s1"]["max_content_jaccard"], 0.0)
        self.assertEqual(rows["ok"]["max_content_jaccard"], 0.0)
        self.assertNotIn("near_duplicate", rows["ok"]["flags"])

    def test_near_duplicates_are_grouped_and_flagged(self):
        a = g.audit_records([rec("n1", GOOD), rec("n2", GOOD + " again"), rec("n3", OTHER)])
        rows = self.by_id(a)
        self.assertIn("near_duplicate", rows["n1"]["flags"])
        self.assertIn("near_duplicate", rows["n2"]["flags"])
        self.assertNotIn("near_duplicate", rows["n3"]["flags"])
        self.assertEqual([sorted(grp) for grp in a["near_groups"]], [["n1", "n2"]])

    def test_median_shares_for_even_and_odd_counts(self):
        odd = g.audit_records([rec(str(i), OTHER, prompt=p) for i, p in enumerate(["", GOOD, OTHER])])
        self.assertAlmostEqual(odd["median_prompt_word_share"],
                               sorted(r["prompt_word_share"] for r in odd["records"])[1], places=3)
        even = g.audit_records([rec("a", OTHER, prompt=OTHER), rec("b", GOOD, prompt="")])
        self.assertEqual(even["median_prompt_word_share"], 0.5)


class ProjectFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_load_project_records_tolerates_missing_dir_junk_and_bad_json(self):
        self.assertEqual(g.load_project_records(self.tmp), [])
        rdir = os.path.join(self.tmp, "records")
        os.makedirs(rdir)
        with open(os.path.join(rdir, "b.json"), "w", encoding="utf-8") as f:
            json.dump(rec("b", GOOD), f)
        with open(os.path.join(rdir, "a.json"), "w", encoding="utf-8") as f:
            json.dump(rec("a", OTHER), f)
        with open(os.path.join(rdir, "broken.json"), "w", encoding="utf-8") as f:
            f.write("{nope")
        with open(os.path.join(rdir, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("ignored")
        os.makedirs(os.path.join(rdir, "dir.json"))  # a directory named like a record: OSError, skipped
        self.assertEqual([r["id"] for r in g.load_project_records(self.tmp)], ["a", "b"])

    def test_audit_project_reads_records_and_passes_options(self):
        rdir = os.path.join(self.tmp, "records")
        os.makedirs(rdir)
        with open(os.path.join(rdir, "a.json"), "w", encoding="utf-8") as f:
            json.dump(rec("a", GOOD), f)
        a = g.audit_project(self.tmp, token_counter={"a": 9})
        self.assertEqual((a["checked"], a["passed_count"]), (1, 1))
        q = g.audit_project(self.tmp, queries={"a": "short"})
        self.assertEqual(q["records"][0]["source"], "custom")


class Drafting(unittest.TestCase):
    def test_nothing_flagged_records_drafter_identity_without_drafting(self):
        class Drafter:
            drafter_id, revision = "d1", "r1"

            def draft(self, paths):
                raise AssertionError("must not be called")

        audit = g.audit_records([rec("ok", GOOD)])
        out = g.draft_flagged_records("/proj", audit, Drafter())
        self.assertEqual((out["drafter_id"], out["drafter_revision"]), ("d1", "r1"))
        bare = g.draft_flagged_records("/proj", g.audit_records([rec("ok", GOOD)]), object())
        self.assertEqual((bare["drafter_id"], bare["drafter_revision"]), (None, None))

    def test_drafter_sees_only_image_paths_and_identity_falls_back_to_name(self):
        seen = []

        class Drafter:
            name = "namedrafter"

            def draft(self, paths):
                seen.append(paths)
                return ["  blind draft  ", None]

        audit = g.audit_records([rec("f1", "too short", prompt="secret prompt text", asset="outputs/f1.png"),
                                 rec("f2", "short", asset="")])
        g.draft_flagged_records("/proj", audit, Drafter())
        self.assertEqual(seen, [[os.path.join("/proj", "outputs/f1.png"), ""]])
        rows = {r["id"]: r for r in audit["flagged"]}
        self.assertEqual(rows["f1"]["florence2_draft"], "blind draft")
        self.assertEqual(rows["f2"]["florence2_draft"], "")
        self.assertEqual((rows["f1"]["author_kind"], rows["f1"]["drafter_id"], rows["f1"]["drafter_revision"]),
                         ("captioner_assisted", "namedrafter", "unknown"))
        self.assertEqual(audit["drafter_id"], "namedrafter")


class Siblings(unittest.TestCase):
    def pool(self, *items):
        return [{"asset_path": p, "dhash": h} for p, h in items]

    def test_nearest_in_same_folder_first_and_capped(self):
        pool = self.pool(("a/t.png", "0"), ("a/1.png", "1"), ("a/2.png", "3"), ("a/3.png", "7"), ("a/4.png", "f"),
                         ("a/5.png", "ff"), ("b/x.png", "0"))
        out = g.select_siblings("a/t.png", pool)
        self.assertEqual(out, ["a/1.png", "a/2.png", "a/3.png", "a/4.png", "a/5.png"])

    def test_other_folders_fill_when_same_folder_is_short(self):
        pool = self.pool(("a/t.png", "0"), ("a/1.png", "1"), ("b/near.png", "1"), ("b/far.png", "ff"),
                         ("c/mid.png", "7"), ("c/more.png", "f"), ("b/most.png", "ffff"))
        out = g.select_siblings("a/t.png", pool)
        self.assertEqual(out[0], "a/1.png")
        self.assertEqual(out[1:], ["b/near.png", "c/mid.png", "c/more.png", "b/far.png"])
        self.assertEqual(len(out), 5)

    def test_unknown_target_empty_paths_and_missing_hashes_use_fallback_hash(self):
        pool = [{"asset_path": "", "dhash": "0"}, {"asset_path": "a/x.png"}, {"asset_path": "a/t.png", "dhash": "0"},
                {"asset_path": "a/y.png"}]
        out = g.select_siblings("a/missing.png", pool)
        self.assertCountEqual(out, ["a/x.png", "a/t.png", "a/y.png"])
        self.assertNotIn("", out)
        self.assertEqual(g.select_siblings("a\\t.png", pool).count("a/t.png"), 0)  # backslashes normalised, target skipped
        self.assertEqual(g.select_siblings("a/t.png", []), [])

    def test_hash_helpers(self):
        self.assertEqual(g._hamming_hex("f0", "0f"), 8)
        self.assertEqual(g._fallback_hash_hex("x"), g._fallback_hash_hex("x"))
        self.assertEqual(len(g._fallback_hash_hex("x")), 64)


class ReviewFiles(unittest.TestCase):
    def test_worksheet_hides_current_text_and_handles_missing_image_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "my proj")
            out = os.path.join(tmp, "out", "nested")
            audit = g.audit_records([
                rec("bad1", "too short", prompt="the secret prompt", asset="outputs/with space/bad1.png"),
                rec("bad2", "short again", asset=""),
                rec("good", GOOD, asset="outputs/good.png", author_kind="human"),
            ])
            audit["flagged"][0]["florence2_draft"] = "a | pipe\nand newline"
            audit["drafter_id"], audit["drafter_revision"] = "dr", "rv"
            jsonl, ws = g.write_review_files(proj, out, audit, dhashes={"outputs\\good.png": "0", "outputs/other.png": "1"})
            with open(ws, encoding="utf-8") as f:
                text = f.read()
            with open(jsonl, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
        self.assertIn("Drafter: `dr` @ `rv`.", text)
        self.assertIn("(no image path)", text)
        self.assertIn("%20", text)  # spaces in links are escaped
        self.assertIn("a \\| pipe and newline", text)
        self.assertIn("Existing descriptions passing model-free gate", text)
        self.assertNotIn("the secret prompt", text)
        self.assertNotIn(GOOD, text)
        self.assertEqual([r["id"] for r in rows], ["bad1", "bad2", "good"])
        self.assertEqual(rows[2]["siblings"], [])        # passing records get no siblings in the jsonl
        self.assertEqual(rows[0]["current_text"], "too short")
        self.assertEqual(rows[2]["author_kind"], "human")

    def test_all_passing_worksheet_has_no_flagged_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = g.audit_records([rec("good", GOOD)])
            _, ws = g.write_review_files(tmp, os.path.join(tmp, "o"), audit)
            with open(ws, encoding="utf-8") as f:
                text = f.read()
        self.assertNotIn("Drafter:", text)
        self.assertIn("## Existing descriptions passing model-free gate", text)


class Types(unittest.TestCase):
    def test_stopword_hash_is_stable_across_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.txt"), os.path.join(tmp, "b.txt")
            with open(a, "wb") as f:
                f.write(b"x\ny\n")
            with open(b, "wb") as f:
                f.write(b"x\r\ny\r\n")
            self.assertEqual(g.stopwords_sha256(a), g.stopwords_sha256(b))
            self.assertEqual(g.load_stopwords(a), frozenset({"x", "y"}))
        self.assertIsInstance(g.STOPWORDS, frozenset)


if __name__ == "__main__":
    unittest.main()
