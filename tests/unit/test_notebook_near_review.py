"""Exercise the actual notebook review cells with tiny local fixtures, never cloud data."""

import csv
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/colab-dataset-preparation.ipynb"


def _cell(tag):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(next(c for c in notebook["cells"] if tag in c.get("metadata", {}).get("tags", []))["source"])


def _pair(a, b, source_a, source_b, score=0.01):
    return {"a": a, "b": b, "split_a": a.split("/")[1], "split_b": b.split("/")[1],
            "source_key_a": source_a, "source_key_b": source_b,
            "same_source_key": source_a == source_b, "dhash_distance": 2,
            "rgb_mean_absolute_difference": score}


def _prepare(tmp_path, pairs, saved_name="drive"):
    prepared, saved = tmp_path / "prepared", tmp_path / saved_name
    prepared.mkdir()
    saved.mkdir()
    for index, name in enumerate(sorted({p[key] for p in pairs for key in ("a", "b")})):
        target = prepared / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 100), (index * 17 % 255, 80, 140)).save(target)
        label = prepared / name.replace("images/", "labels/")
        label = label.with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (prepared / "data.yaml").write_text("names: [bottle, cap, label]\n", encoding="utf-8")
    archive = saved / "prepared-candidate.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for item in prepared.rglob("*"):
            if item.is_file():
                stream.write(item, item.relative_to(prepared))
    (saved / "checksums.json").write_text(json.dumps({
        archive.name: hashlib.sha256(archive.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    (saved / "source-export.zip").write_bytes(b"original fixture source")
    ns = {"ranked_pairs": pairs, "PREPARED": prepared, "SAVED": saved,
          "audit": {"errors": [], "cleaning": {"revision": "reviewed-2026-11-02-v1"},
                    "near_duplicate_candidate_count": len(pairs)},
          "require_cloud": lambda: None}  # Only this synthetic harness bypasses the cloud guard.
    return ns


def _run(ns):
    exec(compile(_cell("near-review-build"), str(NOTEBOOK) + ":9A", "exec"), ns)
    return ns


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _batch_view(ns):
    # Capture real display calls without depending on a notebook frontend locally.
    import IPython.display
    from unittest.mock import patch

    outputs = []
    ns["display"] = outputs.append
    with patch.object(IPython.display, "display", outputs.append):
        exec(compile(_cell("near-review-view"), str(NOTEBOOK) + ":9B", "exec"), ns)
    return outputs


def _write_batch(ns, rows):
    ns["write_review_csv"](ns["BATCH_CSV"], ns["BATCH_FIELDS"], rows)


def _batch_save(ns):
    # Exercise the CSV-only entry point with synthetic inputs. A separate integration
    # test below executes the entire 9C cell, including its real embedded pack.
    ns["REVIEW_UPDATES"] = {}
    _prepare_save_functions(ns)
    ns["apply_prepared_review"]()


def _prepare_save_functions(ns):
    import ast

    tree = ast.parse(_cell("near-review-decisions"))
    assert isinstance(tree.body[-1], ast.Expr)
    assert tree.body[-1].value.func.id == "apply_prepared_review"
    # Load the real functions and payload; callers choose fixture input instead of
    # applying the industrial-watch decisions to unrelated fixture image paths.
    tree.body.pop()
    exec(compile(tree, str(NOTEBOOK) + ":9C-functions", "exec"), ns)


@pytest.fixture
def review(tmp_path):
    a, b, c = "images/train/a.png", "images/val/b.png", "images/test/c.png"
    d, e = "images/train/d.png", "images/test/e.png"
    return _prepare(tmp_path, [_pair(a, b, "source-1", "source-1"),
                               _pair(b, c, "source-1", "source-2"),
                               _pair(d, e, "other-1", "other-2", 0.2)])


def test_all_candidates_grouped_without_inventing_duplicate_decisions(review):
    original_prepared = _snapshot(review["PREPARED"])
    original_saved = _snapshot(review["SAVED"])
    ns = _run(review)
    assert len(ns["review_groups"]) == 2
    assert sorted(len(g["members"]) for g in ns["review_groups"]) == [2, 3]
    assert len(ns["review_pairs"]) == 3  # No invented train-a/test-c edge from transitivity.
    rows = _rows(ns["SAVED"] / "near-duplicate-group-pairs-v1.csv")
    assert len(rows) == 3
    assert {tuple(sorted((r["split_a"], r["split_b"]))) for r in rows} == {
        ("train", "val"), ("test", "val"), ("test", "train"),
    }
    assert sum(r["same_source_key"] == "False" for r in rows) == 2
    assert len(_rows(ns["SAVED"] / "near-duplicate-group-images-v1.csv")) == 5
    assert all(r["decision"] == "pending" for r in _rows(ns["SAVED"] / "near-duplicate-decisions-v1.csv"))
    assert not ns["review_state_file"].exists()
    assert ns["review_progress"]({})["unresolved"] == 3
    assert _snapshot(ns["PREPARED"]) == original_prepared
    for name, content in original_saved.items():
        assert (ns["SAVED"] / name).read_bytes() == content


def test_decisions_survive_rerun_and_invalid_batch_cannot_partially_write(review):
    ns = _run(review)
    first, second, third = list(ns["review_pairs"])
    ns["save_review_decisions"]({first: {"decision": "duplicate", "note": "Same source details."},
                                  second: {"decision": "unsure", "note": "Need a larger image."}})
    before = _snapshot(ns["SAVED"])
    with pytest.raises(ValueError):
        ns["save_review_decisions"]({first: {"decision": "different", "note": "Changed mind."},
                                      "P-missing": {"decision": "duplicate", "note": "Invalid ID."}})
    assert _snapshot(ns["SAVED"]) == before
    with pytest.raises(ValueError):
        ns["save_review_decisions"]({third: {"decision": "duplicate", "note": " "}})
    assert _snapshot(ns["SAVED"]) == before
    ns["ranked_pairs"].reverse()
    _run(ns)
    decisions = ns["load_review_decisions"]()
    assert decisions[first]["decision"] == "duplicate"
    assert decisions[second]["decision"] == "unsure"
    ns["save_review_decisions"]({third: {"decision": "different", "note": "Different scene."}})
    assert ns["review_progress"](ns["load_review_decisions"]()) == {
        "pairs": 3, "images": 5, "groups": 2, "pending": 0,
        "duplicate": 1, "different": 1, "unsure": 1, "unresolved": 1,
        "split_cleaning": "NOT_APPLIED",
    }


@pytest.mark.parametrize("change", ["candidate", "pairs"])
def test_changed_candidate_or_pairs_refuse_to_overwrite_prior_decisions(review, change):
    ns = _run(review)
    first = next(iter(ns["review_pairs"]))
    ns["save_review_decisions"]({first: {"decision": "unsure", "note": "Still reviewing."}})
    if change == "candidate":
        (ns["SAVED"] / "checksums.json").write_text(json.dumps({"prepared-candidate.zip": "a" * 64}))
    else:
        ns["ranked_pairs"][0]["rgb_mean_absolute_difference"] = 0.15
    before = _snapshot(ns["SAVED"])
    with pytest.raises(RuntimeError, match="khác"):
        _run(ns)
    assert _snapshot(ns["SAVED"]) == before


def test_all_gallery_and_pair_pages_render_and_escape_review_notes(tmp_path):
    a = "images/train/source.png"
    pairs = [_pair(a, f"images/{'val' if i % 2 else 'test'}/copy-{i:02}.png", "source", "source")
             for i in range(12)]
    ns = _run(_prepare(tmp_path, pairs))
    first = next(iter(ns["review_pairs"]))
    ns["save_review_decisions"]({first: {"decision": "unsure", "note": "<script>not markup</script>"}})
    displays = []
    ns["display"] = displays.append
    for page in (1, 2, 3):
        ns["show_review_group"](1, image_page=page, pair_page=page)
    galleries = [item for item in displays if isinstance(item, Image.Image)]
    tables = [item.data for item in displays if hasattr(item, "data")]
    assert len(galleries) == 6
    for picture in galleries:
        picture.load()
        assert picture.width in (960, 1040)
    content = "".join(tables)
    for pair_id in ns["review_pairs"]:
        assert pair_id in content
    for name in ns["review_images"]:
        assert name in content
    assert "&lt;script&gt;not markup&lt;/script&gt;" in content
    assert "<script>" not in content
    with pytest.raises(ValueError):
        ns["show_review_group"](1, image_page=4)
    with pytest.raises(ValueError):
        ns["show_review_group"](1, pair_page=4)
    with pytest.raises(ValueError):
        ns["show_review_group"](2)
    _batch_view(ns)
    _batch_save(ns)
    assert len(ns["load_review_decisions"]()) == 1  # The default empty updates confirm nothing.


def test_empty_candidates_and_bad_paths(review, tmp_path):
    # Separate root avoids touching the populated fixture.
    empty = tmp_path / "empty"
    empty.mkdir()
    ns = _run(_prepare(empty, []))
    assert ns["review_groups"] == []
    assert _rows(ns["SAVED"] / "near-duplicate-group-pairs-v1.csv") == []
    _batch_view(ns)
    _batch_save(ns)
    review["ranked_pairs"][0]["a"] = "../outside.png"
    with pytest.raises(ValueError, match="không hợp lệ"):
        _run(review)


def test_cloud_guard_runs_before_reports_are_written(review):
    def reject_local():
        raise RuntimeError("Only Colab")

    review["require_cloud"] = reject_local
    before = _snapshot(review["SAVED"])
    with pytest.raises(RuntimeError, match="Only Colab"):
        _run(review)
    assert _snapshot(review["SAVED"]) == before


def test_batch_exports_every_group_image_and_pair_page_once(tmp_path):
    import base64
    from collections import Counter
    from html import unescape
    from html.parser import HTMLParser
    import io
    import re

    # One group has 3 image pages and 4 pair pages, another has a single pair.
    # This catches duplicated/skipped pages when the two page counts differ.
    a = "images/train/base.png"
    pairs = [_pair(a, f"images/val/copy-{i:02}.png", "one", "one") for i in range(13)]
    pairs.append(_pair("images/train/other.png", "images/test/different.png", "two", "three"))
    ns = _run(_prepare(tmp_path, pairs))
    first = next(iter(ns["review_pairs"]))
    ns["save_review_decisions"]({first: {"decision": "unsure", "note": '<img src=x onerror="bad">'}})
    prepared_before = _snapshot(ns["PREPARED"])
    saved_before = _snapshot(ns["SAVED"])
    outputs = _batch_view(ns)
    report = (ns["SAVED"] / "near-duplicate-all-groups-v1.html").read_text(encoding="utf-8")

    class Tables(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells = []
            self.active = False

        def handle_starttag(self, tag, attrs):
            if tag == "td":
                self.active = True

        def handle_endtag(self, tag):
            if tag == "td":
                self.active = False

        def handle_data(self, data):
            if self.active:
                self.cells.append(data)

    tables = Tables()
    tables.feed(report)
    assert Counter(x for x in tables.cells if x in ns["review_pairs"]) == Counter(ns["review_pairs"].keys())
    paths = re.findall(r'<code>(images/[^<]+)</code>', report)
    assert Counter(map(unescape, paths)) == Counter(ns["review_images"].keys())
    assert report.count('<h2 id="group-') == 2
    encoded_sheets = re.findall(r'src="data:image/jpeg;base64,([^"]+)"', report)
    # 4 image pages + 5 pair pages: each shown once in report and notebook.
    assert len(encoded_sheets) == 9
    notebook_images = [o for o in outputs if getattr(o, "format", None) == "jpeg"]
    assert len(notebook_images) == 9
    for encoded in encoded_sheets:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            image.load()
            assert image.width in (960, 1040)
    assert '&lt;img src=x onerror=&quot;bad&quot;&gt;' in report
    assert '<img src=x onerror="bad">' not in report
    rows = _rows(ns["BATCH_CSV"])
    assert len(rows) == len(ns["review_pairs"])
    assert {row["pair_id"] for row in rows} == set(ns["review_pairs"])
    assert sum(row["decision"] == "pending" for row in rows) == len(pairs) - 1
    assert _snapshot(ns["PREPARED"]) == prepared_before
    for name, content in saved_before.items():
        assert (ns["SAVED"] / name).read_bytes() == content


def test_batch_csv_saves_all_at_once_preserves_input_and_reload(review):
    ns = _run(review)
    _batch_view(ns)
    rows = _rows(ns["BATCH_CSV"])
    for row, choice in zip(rows, ("duplicate", "different", "unsure"), strict=True):
        row.update(decision=choice, note='Reviewed, with a quoted "detail"\nand a newline.')
        # A spreadsheet may uppercase booleans and normalize numeric formatting.
        row["same_source_key"] = row["same_source_key"].upper()
        row["group_number"] = f'{float(row["group_number"]):.1f}'
        row["dhash_distance"] = f'{float(row["dhash_distance"]):.1f}'
        row["rgb_mean_absolute_difference"] = f'{float(row["rgb_mean_absolute_difference"]):.12e}'
    _write_batch(ns, rows)
    input_before = ns["BATCH_CSV"].read_bytes()
    _batch_view(ns)  # Rerendering cannot discard an unsaved decision table.
    assert ns["BATCH_CSV"].read_bytes() == input_before
    assert not ns["review_state_file"].exists()  # Viewing never commits CSV edits.
    _batch_save(ns)
    saved = ns["load_review_decisions"]()
    assert len(saved) == 3
    assert {entry["decision"] for entry in saved.values()} == {"duplicate", "different", "unsure"}
    assert ns["review_progress"](saved)["unresolved"] == 1
    before = ns["review_state_file"].read_bytes()
    _batch_save(ns)
    assert ns["review_state_file"].read_bytes() == before  # Timestamps don't churn.
    assert ns["BATCH_CSV"].read_bytes() == input_before
    _run(ns)
    assert ns["load_review_decisions"]() == saved


@pytest.mark.parametrize("fault", ["missing", "repeat", "unknown", "fingerprint", "path", "decision", "note", "pending-note"])
def test_bad_batch_refuses_all_writes(review, fault):
    ns = _run(review)
    _batch_view(ns)
    rows = _rows(ns["BATCH_CSV"])
    for row in rows:
        row.update(decision="duplicate", note="Inspected matching detail.")
    if fault == "missing":
        rows.pop()
    elif fault == "repeat":
        rows.append(dict(rows[0]))
    elif fault == "unknown":
        rows[-1]["pair_id"] = "P-not-a-candidate"
    elif fault == "fingerprint":
        rows[-1]["fingerprint"] = "a" * 64
    elif fault == "path":
        rows[-1]["b"] = "images/val/unreviewed.png"
    elif fault == "decision":
        rows[-1]["decision"] = "all-are-duplicates"
    elif fault == "note":
        rows[-1]["note"] = " "
    else:
        rows[-1]["decision"] = "pending"
    _write_batch(ns, rows)
    before = _snapshot(ns["SAVED"])
    with pytest.raises(ValueError):
        _batch_save(ns)
    assert _snapshot(ns["SAVED"]) == before


def test_pending_input_preserves_prior_decisions_and_csv_overrides_manual(review):
    ns = _run(review)
    _batch_view(ns)
    first, second, third = list(ns["review_pairs"])
    ns["save_review_decisions"]({first: {"decision": "duplicate", "note": "Prior reviewed result."}})
    _batch_save(ns)  # Previously created pending rows must not clear durable state.
    assert ns["load_review_decisions"]()[first]["decision"] == "duplicate"
    rows = _rows(ns["BATCH_CSV"])
    for row in rows:
        if row["pair_id"] == second:
            row.update(decision="different", note="Corrected via CSV.")
    _write_batch(ns, rows)
    ns["REVIEW_UPDATES"] = {
        second: {"decision": "duplicate", "note": "Old manual input."},
        third: {"decision": "unsure", "note": "Manual real review."},
    }
    _prepare_save_functions(ns)
    ns["apply_prepared_review"]()
    saved = ns["load_review_decisions"]()
    assert saved[second]["decision"] == "different"
    assert saved[third]["decision"] == "unsure"
    assert saved[first]["note"] == "Prior reviewed result."


@pytest.mark.parametrize("tag", ["near-review-view", "near-review-decisions"])
def test_batch_cells_guard_cloud_before_any_write(review, tag):
    ns = _run(review)
    _batch_view(ns)

    def reject_local():
        raise RuntimeError("Only Colab")

    ns["require_cloud"] = reject_local
    before = _snapshot(ns["SAVED"])
    with pytest.raises(RuntimeError, match="Only Colab"):
        exec(compile(_cell(tag), str(NOTEBOOK), "exec"), ns)
    assert _snapshot(ns["SAVED"]) == before


def test_interrupted_export_does_not_publish_partial_report(review, monkeypatch):
    ns = _run(review)
    _batch_view(ns)
    report = ns["SAVED"] / "near-duplicate-all-groups-v1.html"
    before = report.read_bytes()

    def interrupted(*args, **kwargs):
        raise RuntimeError("Interrupted render")

    monkeypatch.setitem(ns, "show_review_group", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted render"):
        ns["export_all_review_groups"]()
    assert report.read_bytes() == before


def _fixture_pack(ns):
    identities = sorted([[key, p["a"], p["b"]] for key, p in ns["review_pairs"].items()])
    rows = [{"pair_id": key, "a": p["a"], "b": p["b"], "group_number": p["group_number"],
             "decision": "unsure", "note": "Synthetic evidence remains inconclusive."}
            for key, p in ns["review_pairs"].items()]
    return {"schema": 1, "revision": "fixture-preview", "candidate_run_id": ns["SAVED"].name,
            "pair_identity_sha256": hashlib.sha256(json.dumps(identities, ensure_ascii=True,
                                                             separators=(",", ":")).encode()).hexdigest(),
            "pair_count": len(rows), "summary": {"unsure": len(rows)}, "decisions": rows}


@pytest.mark.parametrize("fault", ["run", "paths", "missing", "duplicate", "group", "decision", "note", "summary"])
def test_preview_pack_rejects_invalid_batch_before_any_write(review, fault):
    ns = _run(review)
    _batch_view(ns)
    _prepare_save_functions(ns)
    pack = _fixture_pack(ns)
    if fault == "run":
        pack["candidate_run_id"] = "another-candidate"
    elif fault == "paths":
        pack["pair_identity_sha256"] = "f" * 64
    elif fault == "missing":
        pack["decisions"].pop()
    elif fault == "duplicate":
        pack["decisions"][-1] = dict(pack["decisions"][0])
    elif fault == "group":
        pack["decisions"][-1]["group_number"] = 999
    elif fault == "decision":
        pack["decisions"][-1]["decision"] = "auto-clean"
    elif fault == "note":
        pack["decisions"][-1]["note"] = " "
    else:
        pack["summary"] = {"duplicate": len(pack["decisions"])}
    before = _snapshot(ns["SAVED"])
    with pytest.raises(ValueError):
        ns["apply_prepared_review"](pack)
    assert _snapshot(ns["SAVED"]) == before


def test_preview_pack_preserves_recorded_decisions_and_unsure_stays_unresolved(review):
    ns = _run(review)
    _batch_view(ns)
    _prepare_save_functions(ns)
    first = next(iter(ns["review_pairs"]))
    ns["save_review_decisions"]({first: {"decision": "different", "note": "Prior human decision."}})
    prior = ns["load_review_decisions"]()[first]
    pack = _fixture_pack(ns)
    ns["apply_prepared_review"](pack)
    saved = ns["load_review_decisions"]()
    assert saved[first] == prior
    assert ns["review_progress"](saved)["pending"] == 0
    assert ns["review_progress"](saved)["unresolved"] == 2
    assert len(_rows(ns["SAVED"] / "near-duplicate-unresolved-v1.csv")) == 2
    state_before = ns["review_state_file"].read_bytes()
    ns["apply_prepared_review"](pack)
    assert ns["review_state_file"].read_bytes() == state_before


def test_entire_9c_applies_embedded_real_pair_manifest_to_synthetic_runtime(tmp_path):
    # Image pixels are synthetic. This proves binding/persistence, not visual accuracy.
    from collections import Counter

    pack_path = NOTEBOOK.parents[1] / "tests/fixtures/notebook-reviews/saved-preview-2026-11-03-v1.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pairs = []
    for row in pack["decisions"]:
        source_a = Path(row["a"]).name.split(".rf.")[0]
        source_b = Path(row["b"]).name.split(".rf.")[0]
        pairs.append(_pair(row["a"], row["b"], source_a, source_b, row["group_number"] / 1000))
    ns = _run(_prepare(tmp_path, pairs, saved_name=pack["candidate_run_id"]))
    # Load 9B's real CSV parser/column setup without rendering 844 fixture pictures.
    prefix = _cell("near-review-view").split("decisions_for_report = load_review_decisions()", 1)[0]
    exec(compile(prefix, str(NOTEBOOK) + ":9B-csv", "exec"), ns)
    ns["write_review_csv"](ns["BATCH_CSV"], ns["BATCH_FIELDS"], (
        {"fingerprint": ns["review_fingerprint"], **{key: p[key] for key in ns["PAIR_FIELDS"]},
         "decision": "pending", "note": ""} for p in ns["review_pairs"].values()))
    original_prepared = _snapshot(ns["PREPARED"])
    original_saved = _snapshot(ns["SAVED"])
    # Old manual inputs may exist in a live kernel; they must not cause a five-row-only batch.
    first_three = [row for row in pack["decisions"] if row["group_number"] <= 2]
    retained = {row["pair_id"]: {"decision": row["decision"], "note": "Earlier saved human note."}
                for row in first_three}
    ns["save_review_decisions"](retained)
    prior = ns["load_review_decisions"]()
    ns["REVIEW_UPDATES"] = {}
    exec(compile(_cell("near-review-decisions"), str(NOTEBOOK) + ":9C", "exec"), ns)
    assert ns["PREVIEW_REVIEW_PACK"] == pack
    saved = ns["load_review_decisions"]()
    assert len(saved) == 2065
    assert dict(Counter(entry["decision"] for entry in saved.values())) == pack["summary"]
    assert ns["review_progress"](saved)["unresolved"] == pack["summary"]["unsure"]
    assert ns["review_progress"](saved)["pending"] == 0
    assert all(saved[key] == value for key, value in prior.items())
    assert _snapshot(ns["PREPARED"]) == original_prepared
    for name in ("prepared-candidate.zip", "source-export.zip", "checksums.json", ns["BATCH_CSV"].name):
        assert (ns["SAVED"] / name).read_bytes() == original_saved[name]
    state_before = ns["review_state_file"].read_bytes()
    exec(compile(_cell("near-review-decisions"), str(NOTEBOOK) + ":9C", "exec"), ns)
    assert ns["review_state_file"].read_bytes() == state_before


def _split_functions(ns):
    import ast
    tree = ast.parse(_cell("near-review-split-candidate"))
    assert tree.body[-1].targets[0].id == "SPLIT_CANDIDATE_RESULT"
    tree.body.pop()
    exec(compile(tree, str(NOTEBOOK) + ":9D", "exec"), ns)
    return ns


@pytest.fixture
def split_review(tmp_path):
    a, b, c = "images/train/a.png", "images/val/b.png", "images/test/c.png"
    d, e = "images/train/d.png", "images/test/e.png"
    f, g = "images/val/f.png", "images/train/g.png"
    pairs = [_pair(a, b, "ab", "ab"), _pair(b, c, "ab", "c"),
             _pair(d, e, "d", "e"), _pair(f, g, "f", "g")]
    ns = _split_functions(_run(_prepare(tmp_path, pairs)))
    choices = {frozenset((a, b)): "duplicate", frozenset((b, c)): "unsure"}
    ns["save_review_decisions"]({key: {"decision": choices.get(frozenset((p["a"], p["b"])), "different"),
                                      "note": "Synthetic review result."}
                                  for key, p in ns["review_pairs"].items()})
    return ns


def test_9d_creates_separate_candidate_and_preserves_retained_bytes(split_review, tmp_path):
    ns = split_review
    before_saved = _snapshot(ns["SAVED"])
    before_prepared = _snapshot(ns["PREPARED"])
    result = ns["create_split_candidate"](tmp_path / "colab-work")
    report = result["summary"]
    assert report["source_counts"] == {"train": 3, "val": 2, "test": 2}
    assert report["retained_counts"] == {"train": 2, "val": 1, "test": 2}
    assert report["excluded_counts"] == {"train": 1, "val": 1}
    assert report["excluded_total"] == 2
    assert report["parent_unsure_still_unresolved"] == 1
    assert report["training_readiness"] == "NOT_ESTABLISHED"
    assert report["new_dhash_candidate_pairs"] > 0  # Solid fixture pictures all share dHash.
    assert (result["directory"] / "COMPLETE.json").is_file()
    archive = result["directory"] / "prepared-candidate.zip"
    with zipfile.ZipFile(ns["SAVED"] / "prepared-candidate.zip") as parent, zipfile.ZipFile(archive) as output:
        assert "images/train/a.png" not in output.namelist()
        assert "labels/train/a.txt" not in output.namelist()
        assert "images/val/b.png" not in output.namelist()
        assert "labels/val/b.txt" not in output.namelist()
        for name in output.namelist():
            if name.startswith(("images/", "labels/")):
                assert output.read(name) == parent.read(name)
        assert output.testzip() is None
        assert json.loads(output.read("audit.json")) == report
    assert _snapshot(ns["PREPARED"]) == before_prepared
    for name, content in before_saved.items():
        assert (ns["SAVED"] / name).read_bytes() == content
    before = _snapshot(result["directory"])
    repeated = ns["create_split_candidate"](tmp_path / "colab-work")
    assert repeated["directory"] == result["directory"]
    assert _snapshot(result["directory"]) == before


def test_9d_current_preview_policy_matches_offline_counts(split_review):
    pack = json.loads((NOTEBOOK.parents[1] / "tests/fixtures/notebook-reviews/saved-preview-2026-11-03-v1.json").read_text(encoding="utf-8"))
    rows = pack["decisions"]
    pairs = {r["pair_id"]: r for r in rows}
    decisions = {r["pair_id"]: {"decision": r["decision"]} for r in rows}
    plan = split_review["plan_split_blocks"](pairs, decisions)
    from collections import Counter
    assert len(plan["blocks"]) == 173
    assert Counter(name.split("/")[1] for name in plan["exclusions"]) == {"train": 705, "val": 95}


def _rewrite_fixture_archive(ns, mutate):
    archive = ns["SAVED"] / "prepared-candidate.zip"
    with zipfile.ZipFile(archive) as stream:
        entries = {name: stream.read(name) for name in stream.namelist()}
    mutate(entries)
    with zipfile.ZipFile(archive, "w") as stream:
        for name, data in entries.items():
            stream.writestr(name, data)
    ns["candidate_sha"] = hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["missing_label", "invalid_label", "exact_duplicate", "unsafe_path"])
def test_9d_rejects_bad_inputs_before_publication(split_review, tmp_path, fault):
    ns = split_review
    def mutate(entries):
        if fault == "missing_label":
            entries.pop("labels/test/e.txt")
        elif fault == "invalid_label":
            entries["labels/test/e.txt"] = b"0 0.2 0.2 0.5 0.5 0.6 0.6\n"
        elif fault == "exact_duplicate":
            entries["images/test/e.png"] = entries["images/train/d.png"]
        else:
            entries["../outside.txt"] = b"unexpected"
    _rewrite_fixture_archive(ns, mutate)
    before = _snapshot(ns["SAVED"])
    with pytest.raises(ValueError):
        ns["create_split_candidate"](tmp_path / "colab-work")
    assert _snapshot(ns["SAVED"]) == before
    assert not list(ns["SAVED"].rglob("COMPLETE.json"))


def test_9d_changed_parent_and_missing_decisions_block_before_writes(split_review, tmp_path):
    ns = split_review
    archive = ns["SAVED"] / "prepared-candidate.zip"
    with archive.open("ab") as stream:
        stream.write(b"changed")
    before = _snapshot(ns["SAVED"])
    with pytest.raises(RuntimeError, match="gốc thay đổi"):
        ns["create_split_candidate"](tmp_path / "colab-work")
    assert _snapshot(ns["SAVED"]) == before
    ns["load_review_decisions"] = lambda: {}
    with pytest.raises(ValueError, match="đủ"):
        ns["create_split_candidate"](tmp_path / "colab-work")
    assert not (tmp_path / "colab-work").exists()


def test_9d_cloud_guard_and_completed_archive_corruption(split_review, tmp_path):
    ns = split_review
    def reject_local():
        raise RuntimeError("Only Colab")
    original_guard = ns["require_cloud"]
    ns["require_cloud"] = reject_local
    with pytest.raises(RuntimeError, match="Only Colab"):
        ns["create_split_candidate"](tmp_path / "colab-work")
    assert not (tmp_path / "colab-work").exists()
    ns["require_cloud"] = original_guard
    result = ns["create_split_candidate"](tmp_path / "colab-work")
    (result["directory"] / "prepared-candidate.zip").write_bytes(b"corrupted")
    before = _snapshot(result["directory"])
    with pytest.raises(RuntimeError, match="không khớp"):
        ns["create_split_candidate"](tmp_path / "colab-work")
    assert _snapshot(result["directory"]) == before


def test_9d_refuses_empty_evaluation_split(review, tmp_path):
    ns = _split_functions(_run(review))
    ns["save_review_decisions"]({key: {"decision": "unsure", "note": "Unresolved fixture."}
                                  for key in ns["review_pairs"]})
    with pytest.raises(ValueError, match="split bị rỗng"):
        ns["create_split_candidate"](tmp_path / "colab-work")
    assert not list(ns["SAVED"].rglob("COMPLETE.json"))


def test_9d_dhash_index_agrees_with_exhaustive_cross_split_search(split_review):
    values = [0, 1, 3, 7, 15, 31, 1 << 63, (1 << 63) | 1,
              0xABABABABABABABAB, 0xABABABABABABABAC]
    records = [{"image": str(i), "split": ("train", "val", "test")[i % 3], "dhash": value}
               for i, value in enumerate(values)]
    expected = {(a["image"], b["image"], (a["dhash"] ^ b["dhash"]).bit_count())
                for i, b in enumerate(records) for a in records[:i]
                if a["split"] != b["split"] and (a["dhash"] ^ b["dhash"]).bit_count() <= 4}
    actual = split_review["audit_candidate_near_pairs"](records)
    assert {(r["a"], r["b"], r["dhash_distance"]) for r in actual} == expected


def _residual_functions(ns):
    import ast
    tree = ast.parse(_cell("near-review-residual-check"))
    assert tree.body[-1].targets[0].id == "SPLIT_RESIDUAL_CHECK"
    tree.body.pop()
    exec(compile(tree, str(NOTEBOOK) + ":9E", "exec"), ns)
    return ns


def test_9e_recognizes_previously_reviewed_different_warnings(tmp_path):
    a, b, c = "images/train/a.png", "images/val/b.png", "images/test/c.png"
    pairs = [_pair(a, b, "a", "b"), _pair(a, c, "a", "c"), _pair(b, c, "b", "c")]
    ns = _residual_functions(_split_functions(_run(_prepare(tmp_path, pairs))))
    ns["save_review_decisions"]({key: {"decision": "different", "note": "Distinct fixture images."}
                                  for key in ns["review_pairs"]})
    result = ns["create_split_candidate"](tmp_path / "colab-work")
    original = _snapshot(ns["SAVED"])
    summary = ns["check_split_candidate_warnings"](result)
    assert summary["total_dhash_warnings"] == summary["already_reviewed_different"] == 3
    assert summary["requires_review"] == 0
    assert summary["status"] == "ALL_WARNINGS_ALREADY_REVIEWED_DIFFERENT"
    assert summary["decisions_changed_since_9d"] is False
    for name, data in original.items():
        assert (ns["SAVED"] / name).read_bytes() == data


def test_9e_separates_unseen_warnings_and_changed_decisions(split_review, tmp_path):
    ns = _residual_functions(split_review)
    result = ns["create_split_candidate"](tmp_path / "colab-work")
    summary = ns["check_split_candidate_warnings"](result)
    assert summary["already_reviewed_different"] == 2
    assert summary["not_previously_reviewed"] > 0
    assert summary["duplicate_or_unsure"] == 0
    assert summary["status"] == "REVIEW_REQUIRED"
    retained_pair = next(key for key, pair in ns["review_pairs"].items()
                         if pair["a"].endswith("d.png"))
    ns["save_review_decisions"]({retained_pair: {"decision": "unsure", "note": "Reopened fixture review."}})
    updated = ns["check_split_candidate_warnings"](result)
    assert updated["duplicate_or_unsure"] == 1
    assert updated["decisions_changed_since_9d"] is True
    assert updated["requires_review"] == summary["requires_review"] + 1


def test_9e_checksum_and_cloud_guard_stop_before_report_writes(split_review, tmp_path):
    ns = _residual_functions(split_review)
    result = ns["create_split_candidate"](tmp_path / "colab-work")
    (result["directory"] / "prepared-candidate.zip").write_bytes(b"bad archive")
    before = _snapshot(ns["SAVED"])
    with pytest.raises(RuntimeError, match="Checksum"):
        ns["check_split_candidate_warnings"](result)
    assert _snapshot(ns["SAVED"]) == before
    def reject_local():
        raise RuntimeError("Only Colab")
    ns["require_cloud"] = reject_local
    with pytest.raises(RuntimeError, match="Only Colab"):
        ns["check_split_candidate_warnings"](result)
    assert _snapshot(ns["SAVED"]) == before
