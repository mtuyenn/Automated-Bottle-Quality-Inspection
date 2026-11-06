"""Ảnh tổng hợp/ZIP nhỏ: kiểm scoring và lọc cặp; không đo dataset hay train thật."""
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageFilter
import pytest


@pytest.fixture
def blur_env():
    notebook = Path(__file__).parents[2] / "notebooks/colab-blur-filter.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if c.get("id") == "v2-blur-filter")
    tree = ast.parse("".join(cell["source"]))
    tree.body = [n for n in tree.body if not isinstance(n, ast.Expr)]
    env = {}
    exec(compile(tree, str(notebook), "exec"), env)
    return env


@pytest.fixture
def archive(tmp_path, blur_env):
    pixels = ((np.indices((192, 192)).sum(axis=0) // 6 % 2) * 255).astype(np.uint8)
    sharp = Image.fromarray(pixels).convert("RGB")
    blurred = sharp.filter(ImageFilter.GaussianBlur(5))
    pairs, raw_files = [], {}
    # Deliberately not filename-sorted: IDs must follow the audited pair manifest.
    for split, name, image in [("train", "z", sharp), ("train", "a", blurred),
                                ("train", "m", blurred), ("val", "v", sharp), ("test", "t", sharp), ("val", "v2", sharp), ("test", "t2", sharp)]:
        b = io.BytesIO(); image.save(b, format="PNG")
        ib, lb = b.getvalue(), b"0 .5 .5 .9 .9\n"
        ip, lp = f"images/{split}/{name}.png", f"labels/{split}/{name}.txt"
        raw_files.update({ip: ib, lp: lb})
        pairs.append(dict(image=ip, label=lp, split=split, image_sha256=hashlib.sha256(ib).hexdigest(),
                          label_sha256=hashlib.sha256(lb).hexdigest(), original_source="fixture-camera"))
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, raw in raw_files.items():
            z.writestr(name, raw)
        z.writestr("retained-checksums.json", json.dumps(pairs))
        z.writestr("old-summary.json", '{"count":999}')
    digest = blur_env["blur_sha"](path)
    scan = blur_env["blur_scan"](path, tmp_path / "scans", digest, blur_env["BLUR_CONFIG"])
    return path, digest, scan, raw_files


def decision(r, action):
    return {"id": r["id"], "image_sha256": r["image_sha256"], "label_sha256": r["label_sha256"],
            "action": action, "issue": "synthetic_blur", "reason": "synthetic blur fixture", "reviewer_id": "fixture",
            "reviewed_at_utc": "2026-11-06T00:00:00Z"}


def test_real_blur_scoring_and_scan_do_not_exclude(blur_env, archive):
    source, digest, scan, _ = archive
    rows = json.loads((scan / "scores.json").read_text())
    train = [r for r in rows if r["split"] == "train"]
    assert train[0]["image"].endswith("/z.png")
    assert train[0]["full_score"] > train[1]["full_score"]
    assert train[0]["bottle_min_score"] > train[1]["bottle_min_score"]
    assert len(rows) == 7
    assert json.loads((scan / "COMPLETE.json").read_text())["auto_excluded"] == 0
    assert not (scan / "prepared-candidate.zip").exists()
    assert blur_env["blur_scan"](source, scan.parent, digest, blur_env["BLUR_CONFIG"]) == scan


def test_build_all_splits_and_preserves_unresolved_and_source_bytes(blur_env, archive, tmp_path):
    source, digest, scan, raw = archive
    rows = json.loads((scan / "scores.json").read_text())
    ds = [decision(r, "exclude_quality" if i in {1, 3, 4} else "unresolved" if i == 2 else "keep") for i, r in enumerate(rows)]
    removed = {rows[i][field] for i in (1, 3, 4) for field in ("image", "label")}
    dp = scan / "decisions-reviewed.json"
    dp.write_text(json.dumps({"source_sha256": digest, "decisions": ds}))
    out = blur_env["blur_build"](source, scan, dp, tmp_path / "candidates", digest)
    assert blur_env["blur_sha"](source) == digest
    with zipfile.ZipFile(out / "prepared-candidate.zip") as z:
        assert rows[1]["image"] not in z.namelist()
        assert rows[1]["label"] not in z.namelist()
        for name, value in raw.items():
            if name not in removed:
                assert z.read(name) == value
        pairs = json.loads(z.read("retained-checksums.json"))
        assert len(pairs) == 4 and all(p["original_source"] == "fixture-camera" for p in pairs)
        assert "old-summary.json" not in z.namelist()
        assert "2: label" in z.read("data.yaml").decode()
    report = json.loads((out / "COMPLETE.json").read_text())
    assert report["counts"] == {"train": 2, "val": 1, "test": 1}
    assert report["training_ready"] is False
    assert report["unresolved_ids"] == [rows[2]["id"]]
    assert report["reviewed_count"] == 7
    assert report["evaluation_split_changed"] is True


def test_review_next_page_uses_pinned_config_and_source(blur_env, archive, tmp_path):
    source, digest, _, _ = archive
    config = dict(blur_env["BLUR_CONFIG"], review_page_size=1)
    scan = blur_env["blur_scan"](source, tmp_path / "small-pages", digest, config)
    order = json.loads((scan / "review-order.json").read_text())
    page, template = blur_env["blur_review_page"](source, scan, 2)
    review = json.loads(template.read_text())
    assert review["source_sha256"] == digest
    assert [d["id"] for d in review["decisions"]] == [order[1]]
    assert len(list(page.glob("*-source.png"))) == 1
    assert blur_env["blur_review_page"](source, scan, 2) == (page, template)
    assert blur_env["blur_scan"](source, scan.parent, digest, config) == scan


@pytest.mark.parametrize("invalid", ["missing", "pending_original", "hash", "duplicate", "no_review", "no_issue"])
def test_build_rejects_invalid_decisions(blur_env, archive, tmp_path, invalid):
    source, digest, scan, _ = archive
    rows = json.loads((scan / "scores.json").read_text())
    ds = [decision(r, "exclude_quality" if i == 1 else "keep") for i, r in enumerate(rows)]
    if invalid == "missing":
        ds.pop()
    elif invalid == "pending_original":
        ds[0]["action"] = "pending_original"
    elif invalid == "no_issue":
        ds[1].pop("issue")
    elif invalid == "hash":
        ds[0]["image_sha256"] = "0" * 64
    elif invalid == "duplicate":
        ds.append(dict(ds[0]))
    elif invalid == "no_review":
        ds[0]["reviewer_id"] = ""
    dp = scan / "decisions-reviewed.json"
    dp.write_text(json.dumps({"source_sha256": digest, "decisions": ds}))
    with pytest.raises(ValueError):
        blur_env["blur_build"](source, scan, dp, tmp_path / "candidates", digest)
    assert not (tmp_path / "candidates").exists()
