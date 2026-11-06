"""Exercise source-bound repair exports and timing mechanics, without real training or GPU claims."""
import ast
import hashlib
import io
import json
import math
import random
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest
from PIL import Image


NOTEBOOK = Path(__file__).parents[2] / "notebooks/colab-training-preparation.ipynb"


def cell_functions():
    env = {}
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if any(t in cell.get("metadata", {}).get("tags", []) for t in ("training-v2-repair", "training-speed")):
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), env)
    return env


def v2_functions():
    """Load only imports, constants and function definitions from unexecuted v2 cells."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tags = {
        "training-v2-core", "training-v2-audit", "training-v2-semantic-review",
        "training-v2-ingestion", "training-v2-repair-attestation", "training-v2-candidate",
        "training-v2-training", "training-v2-evaluation", "training-v2-timing",
    }
    env = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or not tags.intersection(cell.get("metadata", {}).get("tags", [])):
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), env)
    return env


def _authorized_fixture(tmp_path, env, *, target_present=True, label_bytes=b"0 .5 .5 .4 .6\n"):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    image_path = inbox / "source.jpg"
    stream = io.BytesIO()
    Image.new("RGB", (100, 80), (25, 75, 125)).save(stream, format="JPEG")
    image_bytes = stream.getvalue()
    image_path.write_bytes(image_bytes)
    label_path = inbox / "source.txt"
    label_path.write_bytes(label_bytes)
    record = {
        "id": "source-001", "image": image_path.name, "label": label_path.name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "target_present": target_present, "hard_negative": False,
        "annotation_policy_revision": env["V2_POLICY_REVISION"],
        "annotation_review": {
            "complete": True, "visible_instances_complete": True,
            "visible_instance_count": len(label_bytes.splitlines()) if label_bytes.strip() else 0,
            "reviewer_id": "reviewer-fixture", "reviewed_at_utc": "2026-11-05T12:00:00Z",
        },
        "risk_tags": ["machine_background"],
    }
    manifest = {
        "source_set_id": "authorized-fixture-v1",
        "rights": {
            "rights_confirmed": True, "training_use": True, "license": "fixture-authorized",
            "authorization_ref": "fixture-consent-1", "confirmed_by": "owner",
            "rights_confirmed_at_utc": "2026-11-05T11:00:00Z",
        },
        "records": [record],
    }
    manifest_path = inbox / "authorized-source-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return inbox, manifest_path, manifest, record, image_bytes, label_bytes


def _candidate_build_fixture(tmp_path, env, *, duplicate_across_splits=False):
    drive = tmp_path / "drive"
    drive.mkdir()
    audit = drive / "audit"
    audit.mkdir()
    intake = drive / "intake"
    intake.mkdir()
    receipts = drive / "receipts"
    receipts.mkdir()
    parent_zip = drive / "parent.zip"

    sources = []
    groups = [
        ("train", "g-train", ("machine_background", "motion_blur", "occlusion", "hard_negative"), False),
        ("train", "g-train", (), True),
        ("val", "g-val", ("reflection", "small_cap", "physical_sleeve"), True),
        ("test", "g-test", ("partial_bottle",), True),
    ]
    records = []
    inventory = []
    semantic = []
    legacy_groups = {}
    group_splits = {"g-train": "train", "g-val": "val", "g-test": "test"}
    first_image_bytes = None
    with zipfile.ZipFile(parent_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (split, group_id, risk_tags, target_present) in enumerate(groups):
            image_name = f"images/{split}/source-{index}.jpg"
            label_name = f"labels/{split}/source-{index}.txt"
            if duplicate_across_splits and index == 3:
                image_bytes = first_image_bytes
            else:
                image = Image.new("L", (128, 96), 255)
                pixels = image.load()
                rng = random.Random(700 + index)
                for y in range(12):
                    for x in range(16):
                        value = 0 if rng.randrange(2) else 255
                        for dy in range(8):
                            for dx in range(8):
                                pixels[x * 8 + dx, y * 8 + dy] = value
                stream = io.BytesIO()
                image.save(stream, format="JPEG")
                image_bytes = stream.getvalue()
                if index == 0:
                    first_image_bytes = image_bytes
            label_bytes = b"0 .5 .5 .4 .6\n" if target_present else b""
            archive.writestr(image_name, image_bytes)
            archive.writestr(label_name, label_bytes)
            row = {
                "split": split, "image": image_name, "label": label_name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
            }
            records.append(row)
            meta = env["v2_image_metadata"](image_bytes)
            boxes = env["v2_parse_yolo"](label_bytes.decode(), meta["width"], meta["height"])
            ident = f"{split}-{index + 1:06d}"
            counts = {name: sum(box["class_name"] == name for box in boxes)
                      for name in env["V2_CLASSES"].values()}
            inventory.append({
                "id": ident, "split": split, "image": image_name, "label": label_name,
                "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
                "width": meta["width"], "height": meta["height"], "structurally_valid": True,
                "class_counts": counts,
            })
            semantic.append({
                "id": ident, "status": "reviewed", "visible_instances_complete": True,
                "policy_revision": env["V2_POLICY_REVISION"], "reviewer_id": "full-reviewer",
                "reviewed_at_utc": "2026-11-05T13:00:00Z", "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"], "target_present": target_present,
                "hard_negative": not target_present, "no_target_confirmed": not target_present,
                "final_yolo": label_bytes.decode(),
            })
            if not (receipts / f"{group_id}.json").exists():
                group_file = receipts / f"{group_id}.json"
                group_doc = {
                    "group_id": group_id, "capture_type": "camera_session", "verified": True,
                    "rights_confirmed": True, "training_use": True, "license": "fixture-authorized",
                    "authorization_ref": "session-consent-fixture",
                    "reviewer_id": "session-reviewer", "reviewed_at_utc": "2026-11-05T12:30:00Z",
                    "risk_tags": list(risk_tags),
                }
                group_file.write_text(json.dumps(group_doc), encoding="utf-8")
                legacy_groups[image_name] = {
                    "group_id": group_id, "capture_type": "camera_session", "verified": True,
                    "group_manifest_relative": f"receipts/{group_id}.json",
                    "group_manifest_sha256": hashlib.sha256(group_file.read_bytes()).hexdigest(),
                    "risk_tags": list(risk_tags),
                }
            else:
                legacy_groups[image_name] = {
                    "group_id": group_id, "capture_type": "camera_session", "verified": True,
                    "group_manifest_relative": f"receipts/{group_id}.json",
                    "group_manifest_sha256": hashlib.sha256((receipts / f"{group_id}.json").read_bytes()).hexdigest(),
                    "risk_tags": [],
                }
        archive.writestr("retained-checksums.json", json.dumps(records))
        archive.writestr("data.yaml", "path: .\nnames:\n  0: bottle\n  1: cap\n  2: label\n")

    parent_sha = hashlib.sha256(parent_zip.read_bytes()).hexdigest()
    (audit / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    env["v2_atomic_json"](audit / "COMPLETE.json", {
        "source_zip_sha256": parent_sha,
        "files": {"inventory.json": hashlib.sha256((audit / "inventory.json").read_bytes()).hexdigest()},
    })
    semantic_path = audit / "semantic-decisions.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    coverage = {
        "reviewed_count": len(semantic), "unreviewed_count": 0, "unresolved_count": 0,
        "reviewed_ids": [row["id"] for row in semantic], "unreviewed_ids": [], "unresolved_ids": [],
    }
    semantic_receipt = {
        "audit_source_zip_sha256": parent_sha, "policy_revision": env["V2_POLICY_REVISION"],
        "decisions_sha256": hashlib.sha256(semantic_path.read_bytes()).hexdigest(), "coverage": coverage,
    }
    semantic_receipt_path = receipts / "semantic.json"
    semantic_receipt_path.write_text(json.dumps(semantic_receipt), encoding="utf-8")

    intake_payload = {
        "source_manifest_sha256": "fixture-intake-manifest", "rights": {
            "rights_confirmed": True, "training_use": True, "license": "fixture-authorized",
            "authorization_ref": "fixture-consent", "confirmed_by": "owner",
        }, "accepted": [], "quarantine": [], "missing_risk_tags": [], "candidate_ready": False,
    }
    intake_json = intake / "ingestion.json"
    intake_json.write_text(json.dumps(intake_payload), encoding="utf-8")
    intake_marker = {
        "source_manifest_sha256": "fixture-intake-manifest",
        "files": {"ingestion.json": hashlib.sha256(intake_json.read_bytes()).hexdigest()},
    }
    (intake / "COMPLETE.json").write_text(json.dumps(intake_marker), encoding="utf-8")

    reviewed_path = receipts / "bounded-decisions.json"
    reviewed_path.write_text(json.dumps({
        "review_status": "PIXEL_REJUDGED_BOUNDED_SAMPLE", "candidate_authorized": False, "decisions": [],
    }), encoding="utf-8")
    attestation_path = receipts / "repair-attestation.json"
    attestation_path.write_text(json.dumps({
        "policy_revision": env["V2_POLICY_REVISION"],
        "reviewed_artifact_sha256": hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
        "candidate_authorized": True, "full_candidate_review_complete": False,
        "reviewer_id": "independent", "reviewed_at_utc": "2026-11-05T13:30:00Z", "decisions": {},
    }), encoding="utf-8")
    receipt_docs = {
        "rights": {"source_zip_sha256": parent_sha, "rights_confirmed": True,
                   "training_use": True, "license": "fixture-authorized", "confirmed_by": "owner",
                   "rights_confirmed_at_utc": "2026-11-05T13:30:00Z"},
        "policy": {"policy_revision": env["V2_POLICY_REVISION"], "approved": True,
                   "reviewer_id": "policy-reviewer", "reviewed_at_utc": "2026-11-05T13:35:00Z"},
        "static": {"static_review_complete": True, "execution_claimed": False,
                   "reviewer_id": "static-reviewer", "reviewed_at_utc": "2026-11-05T13:40:00Z"},
        "holdout": {"independent": True, "frozen_before_tuning": True, "tuning_started": False,
                    "holdout_group_ids": ["g-test"], "reviewer_id": "holdout-reviewer",
                    "reviewed_at_utc": "2026-11-05T13:45:00Z"},
    }
    for name, document in receipt_docs.items():
        (receipts / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    def rel_and_sha(path):
        return str(path.relative_to(drive)).replace("\\", "/"), hashlib.sha256(path.read_bytes()).hexdigest()
    build = {
        "candidate_id": "candidate-v2-fixture-01", "source_intake_relative": "intake",
        "group_splits": group_splits, "legacy_groups": legacy_groups,
    }
    path_map = {
        "rights_receipt": receipts / "rights.json", "policy_receipt": receipts / "policy.json",
        "static_review_receipt": receipts / "static.json", "holdout_freeze_receipt": receipts / "holdout.json",
        "repair_attestation": attestation_path, "reviewed_decisions": reviewed_path,
        "semantic_review_receipt": semantic_receipt_path, "semantic_decisions": semantic_path,
    }
    for key, path in path_map.items():
        relative, digest = rel_and_sha(path)
        build[key + "_relative"] = relative
        build[key + "_sha256"] = digest
    return {
        "drive": drive, "audit": audit, "intake": intake, "parent_zip": parent_zip,
        "parent_sha": parent_sha, "build": build, "output": drive / "candidate-v2",
        "source_bytes": parent_zip.read_bytes(), "inventory": inventory,
    }


def test_v2_parser_drawing_and_partial_box_edges_are_hand_computable():
    env = v2_functions()
    box = env["v2_parse_yolo"]("0 0.5 0.5 0.4 0.6", 100, 80)[0]
    assert box["xyxy"] == [30.0, 16.0, 70.0, 64.0]
    image = Image.new("RGB", (100, 80), (40, 40, 40))
    rendered, _ = env["v2_draw_yolo"](image, "0 0.5 0.5 0.4 0.6")
    assert rendered.getpixel((30, 40)) == (35, 105, 255)
    assert rendered.getpixel((69, 63)) == (35, 105, 255)
    assert rendered.getpixel((50, 40)) == (40, 40, 40)
    with pytest.raises(ValueError, match="outside image"):
        env["v2_parse_yolo"]("0 0.9 0.5 0.4 0.2", 100, 80)


def test_v2_path_hash_copy_and_source_preservation(tmp_path):
    env = v2_functions()
    source = tmp_path / "source.bin"
    destination = tmp_path / "stage" / "copy.bin"
    source.write_bytes(b"source bytes")
    before = env["v2_sha256"](source)
    assert env["v2_copy_verified"](source, destination, before) == before
    assert env["v2_sha256"](source) == before
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError):
        env["v2_resolve_below"](tmp_path, "../escape.bin")
    with pytest.raises(RuntimeError):
        env["v2_copy_verified"](source, tmp_path / "bad.bin", "0" * 64)


def test_semantic_coverage_keeps_reviewed_unreviewed_and_unresolved_separate():
    env = v2_functions()
    coverage = env["v2_review_coverage"]({"reviewed", "pending", "uncertain"}, [
        {"id": "reviewed", "status": "reviewed"},
        {"id": "uncertain", "status": "unresolved"},
    ])
    assert coverage["reviewed_ids"] == ["reviewed"]
    assert coverage["unresolved_ids"] == ["uncertain"]
    assert coverage["unreviewed_ids"] == ["pending"]
    assert coverage["semantic_clean_claim"] is False
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        env["v2_review_coverage"]({"one"}, [
            {"id": "one", "status": "reviewed"}, {"id": "one", "status": "reviewed"},
        ])


def test_semantic_batch_is_idempotent_and_rejects_target_bearing_empty_final_label(tmp_path):
    env = v2_functions()
    archive = tmp_path / "source.zip"
    stream = io.BytesIO()
    Image.new("RGB", (100, 80), (90, 100, 110)).save(stream, format="PNG")
    image_bytes = stream.getvalue()
    original_label = b"0 .5 .5 .2 .2\n"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("images/train/a.png", image_bytes)
        z.writestr("labels/train/a.txt", original_label)
    source_sha = env["v2_sha256"](archive)
    audit = tmp_path / "audit"
    audit.mkdir()
    inventory = [{
        "id": "train-000001", "split": "train", "image": "images/train/a.png",
        "label": "labels/train/a.txt", "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "label_sha256": hashlib.sha256(original_label).hexdigest(), "width": 100, "height": 80,
    }]
    inventory_path = audit / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    (audit / "COMPLETE.json").write_text(json.dumps({
        "source_zip_sha256": source_sha,
        "files": {"inventory.json": env["v2_sha256"](inventory_path)},
    }), encoding="utf-8")
    decisions_path = audit / "semantic-decisions.json"
    first_pack, first_receipt, first_receipt_path = env["v2_semantic_batch"](
        audit, archive, decisions_path, batch_size=1,
    )
    second_pack, second_receipt, second_receipt_path = env["v2_semantic_batch"](
        audit, archive, decisions_path, batch_size=1,
    )
    assert first_pack == second_pack and first_receipt == second_receipt
    assert first_receipt_path == second_receipt_path
    assert (first_pack / "COMPLETE.json").is_file()

    decision = {
        **inventory[0], "status": "reviewed", "policy_revision": env["V2_POLICY_REVISION"],
        "reviewer_id": "pixel-reviewer", "reviewed_at_utc": "2026-11-05T12:00:00Z",
        "visible_instances_complete": True, "target_present": True, "hard_negative": False,
        "no_target_confirmed": False,
        "final_yolo": "0 0.5 0.5 1.0 0.5\n2 0.5 0.55 0.4 0.2\n",
    }
    decisions_path.write_text(json.dumps([decision]), encoding="utf-8")
    _, reviewed_receipt, _ = env["v2_semantic_batch"](audit, archive, decisions_path, batch_size=1)
    assert reviewed_receipt["coverage"]["reviewed_count"] == 1
    assert reviewed_receipt["coverage"]["unresolved_count"] == 0
    decision["final_yolo"] = ""
    decisions_path.write_text(json.dumps([decision]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Target-bearing image"):
        env["v2_semantic_batch"](audit, archive, decisions_path, batch_size=1)


def test_authorized_intake_rejects_unknown_rights_before_opening_source(tmp_path):
    env = v2_functions()
    manifest = {"rights": {"rights_confirmed": False}, "records": [
        {"image": "missing.jpg", "label": "missing.txt"},
    ]}
    with pytest.raises(PermissionError, match="Rights"):
        env["v2_validate_authorized_source_manifest"](manifest, tmp_path)


def test_authorized_intake_rejects_target_bearing_empty_label(tmp_path):
    env = v2_functions()
    inbox, _, manifest, _, _, _ = _authorized_fixture(
        tmp_path, env, target_present=True, label_bytes=b"",
    )
    with pytest.raises(RuntimeError, match="empty label"):
        env["v2_validate_authorized_source_manifest"](manifest, inbox)


def test_authorized_intake_quarantines_unknown_group_and_reuses_hash_bound_result(tmp_path):
    env = v2_functions()
    inbox, manifest_path, manifest, _, image_bytes, label_bytes = _authorized_fixture(tmp_path, env)
    image_before = (inbox / "source.jpg").read_bytes()
    label_before = (inbox / "source.txt").read_bytes()
    checked = env["v2_validate_authorized_source_manifest"](manifest, inbox)
    assert checked["status"] == "QUARANTINE_PRESENT"
    assert checked["records"][0]["group_known"] is False
    output_root = tmp_path / "intake-output"
    target, marker = env["v2_ingest_authorized_sources"](manifest_path, inbox, output_root)
    ingestion = json.loads((target / "ingestion.json").read_text(encoding="utf-8"))
    assert marker["status"] == "QUARANTINE_PRESENT"
    assert ingestion["accepted"] == []
    assert len(ingestion["quarantine"]) == 1
    assert (target / ingestion["quarantine"][0]["image"]).read_bytes() == image_bytes
    assert (target / ingestion["quarantine"][0]["label"]).read_bytes() == label_bytes
    again, again_marker = env["v2_ingest_authorized_sources"](manifest_path, inbox, output_root)
    assert again == target and again_marker == marker
    assert (inbox / "source.jpg").read_bytes() == image_before
    assert (inbox / "source.txt").read_bytes() == label_before


def test_candidate_build_is_source_preserving_idempotent_and_refuses_manifest_conflict(tmp_path):
    env = v2_functions()
    fixture = _candidate_build_fixture(tmp_path, env)
    args = (fixture["drive"], fixture["parent_zip"], fixture["parent_sha"], fixture["audit"],
            fixture["intake"], fixture["build"], fixture["output"])
    target, marker = env["v2_build_grouped_candidate"](*args)
    assert marker["status"] == "CANDIDATE_V2_COMPLETE"
    assert (target / "prepared-candidate.zip").is_file()
    assert json.loads((target / "training-ready-receipt.json").read_text())["ready"] is True
    assert fixture["parent_zip"].read_bytes() == fixture["source_bytes"]
    again, again_marker = env["v2_build_grouped_candidate"](*args)
    assert again == target and again_marker == marker
    conflict = dict(fixture["build"], review_note="different binding")
    with pytest.raises(FileExistsError, match="conflicts"):
        env["v2_build_grouped_candidate"](
            fixture["drive"], fixture["parent_zip"], fixture["parent_sha"], fixture["audit"],
            fixture["intake"], conflict, fixture["output"],
        )
    assert fixture["parent_zip"].read_bytes() == fixture["source_bytes"]


def test_candidate_build_quarantines_duplicate_connected_split_leakage(tmp_path):
    env = v2_functions()
    fixture = _candidate_build_fixture(tmp_path, env, duplicate_across_splits=True)
    with pytest.raises(RuntimeError, match="Duplicate-connected"):
        env["v2_build_grouped_candidate"](
            fixture["drive"], fixture["parent_zip"], fixture["parent_sha"], fixture["audit"],
            fixture["intake"], fixture["build"], fixture["output"],
        )
    assert not fixture["output"].exists()
    assert fixture["parent_zip"].read_bytes() == fixture["source_bytes"]


def test_training_readiness_receipts_fail_closed_and_bind_candidate(tmp_path):
    env = v2_functions()
    drive = tmp_path / "drive"
    candidate = drive / "candidate"
    receipts = drive / "receipts"
    receipts.mkdir(parents=True)
    candidate.mkdir(parents=True)
    archive = candidate / "prepared-candidate.zip"
    archive.write_bytes(b"candidate fixture")
    candidate_sha = env["v2_sha256"](archive)
    parent_sha = "p" * 64
    decision_path = receipts / "semantic-decisions.json"
    decisions = [{"id": "reviewed-pair-001", "status": "reviewed"}]
    decision_path.write_text(json.dumps(decisions), encoding="utf-8")
    decision_sha = env["v2_sha256"](decision_path)
    documents = {
        "rights_receipt": {
            "source_zip_sha256": parent_sha, "rights_confirmed": True, "training_use": True,
            "license": "fixture-authorized", "confirmed_by": "owner",
            "rights_confirmed_at_utc": "2026-11-05T12:00:00Z",
        },
        "policy_receipt": {
            "policy_revision": "policy-fixture", "approved": True,
            "reviewer_id": "policy-reviewer", "reviewed_at_utc": "2026-11-05T12:01:00Z",
        },
        "static_review_receipt": {
            "static_review_complete": True, "execution_claimed": False,
            "reviewer_id": "static-reviewer", "reviewed_at_utc": "2026-11-05T12:02:00Z",
        },
        "holdout_freeze_receipt": {
            "independent": True, "frozen_before_tuning": True, "tuning_started": False,
            "holdout_group_ids": ["group-test"], "reviewer_id": "holdout-reviewer",
            "reviewed_at_utc": "2026-11-05T12:03:00Z",
        },
        "semantic_review_receipt": {
            "audit_source_zip_sha256": parent_sha, "policy_revision": "policy-fixture",
            "decisions_sha256": decision_sha, "semantic_clean_claim": False,
            "coverage": {"reviewed_count": 1, "unreviewed_count": 0, "unresolved_count": 0},
        },
    }
    ready = {
        "ready": True, "candidate_id": "candidate-v2-fixture", "candidate_zip_sha256": candidate_sha,
        "parent_zip_sha256": parent_sha, "policy_revision": "policy-fixture",
        "candidate_readback_verified": True, "candidate_created": True,
        "full_annotation_review_complete": True, "rights_confirmed": True,
        "holdout_frozen_before_tuning": True, "static_review_complete": True,
        "semantic_decisions_relative": "receipts/semantic-decisions.json",
        "semantic_decisions_sha256": decision_sha,
    }
    for name, document in documents.items():
        receipt = receipts / (name + ".json")
        receipt.write_text(json.dumps(document), encoding="utf-8")
        ready[name + "_relative"] = f"receipts/{receipt.name}"
        ready[name + "_sha256"] = env["v2_sha256"](receipt)
    ready_path = candidate / "training-ready-receipt.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    marker = {
        "status": "CANDIDATE_V2_COMPLETE", "parent_zip_sha256": parent_sha,
        "candidate_zip_sha256": candidate_sha,
        "files": {"prepared-candidate.zip": env["v2_sha256"](archive),
                  "training-ready-receipt.json": env["v2_sha256"](ready_path)},
    }
    marker_path = candidate / "COMPLETE.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    result = env["v2_validate_candidate_readiness"](
        candidate, drive, parent_sha, "policy-fixture", "candidate-v2-fixture",
    )
    assert result[0] == candidate_sha
    ready["static_review_complete"] = False
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    marker["files"]["training-ready-receipt.json"] = env["v2_sha256"](ready_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(PermissionError, match="readiness gates"):
        env["v2_validate_candidate_readiness"](
            candidate, drive, parent_sha, "policy-fixture", "candidate-v2-fixture",
        )
    ready["static_review_complete"] = True
    holdout_path = receipts / "holdout_freeze_receipt.json"
    documents["holdout_freeze_receipt"]["independent"] = False
    holdout_path.write_text(json.dumps(documents["holdout_freeze_receipt"]), encoding="utf-8")
    ready["holdout_freeze_receipt_sha256"] = env["v2_sha256"](holdout_path)
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    marker["files"]["training-ready-receipt.json"] = env["v2_sha256"](ready_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(PermissionError, match="Independent holdout"):
        env["v2_validate_candidate_readiness"](
            candidate, drive, parent_sha, "policy-fixture", "candidate-v2-fixture",
        )

def test_training_identity_binds_explicit_candidate_path_separately_from_run_name(tmp_path):
    env = v2_functions()
    run_name = "bottleqc-yolo11n-640-v2-independent-run"
    candidate_id = "candidate-v2-arbitrary-source"
    candidate_relative = f"BottleQC/datasets/candidate-v2/{candidate_id}"
    config = {"candidate_id": candidate_id, "candidate_relative": candidate_relative}
    assert run_name != candidate_id
    candidate_path = env["v2_validate_training_identity"](
        run_name, candidate_id, candidate_relative, tmp_path,
    )
    assert candidate_path == tmp_path / candidate_relative
    binding = env["v2_checkpoint_binding"](
        run_name, candidate_id, candidate_relative, "candidate-sha", config, "v1-sha",
    )
    assert binding["run_name"] == run_name
    assert binding["candidate_id"] == candidate_id
    assert binding["candidate_relative"] == candidate_relative
    assert binding["config"]["candidate_id"] == candidate_id
    assert binding["config"]["candidate_relative"] == candidate_relative
    with pytest.raises(ValueError, match="Candidate path"):
        env["v2_validate_training_identity"](
            run_name, candidate_id, "BottleQC/datasets/candidate-v2/candidate-v2-other", tmp_path,
        )
    with pytest.raises(ValueError, match="candidate binding mismatch"):
        env["v2_checkpoint_binding"](
            run_name, candidate_id, candidate_relative, "candidate-sha",
            dict(config, candidate_relative="BottleQC/datasets/candidate-v2/other"), "v1-sha",
        )

def test_resume_snapshot_binds_last_optimizer_candidate_and_rejects_completed_or_stripped(tmp_path):
    env = v2_functions()
    run_name = "bottleqc-yolo11n-640-v2-independent-run"
    candidate_id = "candidate-v2-reviewed-source-a"
    candidate_relative = f"BottleQC/datasets/candidate-v2/{candidate_id}"
    config = {"imgsz": 640, "candidate_id": candidate_id, "candidate_relative": candidate_relative}
    assert run_name != candidate_id
    assert env["v2_validate_training_identity"](run_name, candidate_id, candidate_relative, tmp_path) == tmp_path / candidate_relative
    with pytest.raises(ValueError, match="Candidate path"):
        env["v2_validate_training_identity"](run_name, candidate_id, "BottleQC/datasets/candidate-v2/candidate-v2-other", tmp_path)
    with pytest.raises(ValueError, match="candidate binding mismatch"):
        env["v2_checkpoint_binding"](run_name, candidate_id, candidate_relative, "candidate-sha", dict(config, candidate_relative="other"), "v1-sha")
    run = tmp_path / run_name
    checkpoint = run / "checkpoints" / "epoch-0001"
    checkpoint.mkdir(parents=True)
    files = {
        "last.pt": b"last checkpoint with optimizer",
        "best.pt": b"stripped best checkpoint",
        "optimizer-state.pt": b"separate optimizer state",
    }
    digests = {}
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
        digests[name] = env["v2_sha256"](checkpoint / name)
    binding = env["v2_checkpoint_binding"](run_name, candidate_id, candidate_relative, "candidate-sha", config, "v1-sha")
    snapshot = {
        "epoch": 1, "folder": "checkpoints/epoch-0001", "files": digests,
        "binding": binding, "resume_role": "last",
        "optimizer_state_sha256": digests["optimizer-state.pt"], "training_complete": False,
    }
    (checkpoint / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (run / "LATEST.json").write_text(json.dumps(snapshot), encoding="utf-8")
    fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: {"state": {"momentum": 1}})
    assert env["v2_validate_resume_snapshot"](run, binding, fake_torch) == snapshot
    bad_binding = dict(binding, candidate_id="candidate-v2-other")
    with pytest.raises(RuntimeError, match="belongs to another"):
        env["v2_validate_resume_snapshot"](run, bad_binding, fake_torch)
    (checkpoint / "last.pt").write_bytes(files["best.pt"])
    digests["last.pt"] = env["v2_sha256"](checkpoint / "last.pt")
    snapshot["files"] = digests
    (checkpoint / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (run / "LATEST.json").write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stripped best"):
        env["v2_validate_resume_snapshot"](run, binding, fake_torch)
    (run / "TRAIN_COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Completed"):
        env["v2_validate_resume_snapshot"](run, binding, fake_torch)


def test_pair_metric_contract_has_hand_computable_counts_and_denominators():
    env = v2_functions()
    row = {
        "ground_truth": [{"class_id": 0, "xyxy": [10, 10, 30, 30]},
                         {"class_id": 1, "xyxy": [40, 40, 50, 50]}],
        "predictions": [{"class_id": 0, "confidence": 0.9, "xyxy": [10, 10, 30, 30]},
                        {"class_id": 0, "confidence": 0.8, "xyxy": [60, 60, 70, 70]},
                        {"class_id": 2, "confidence": 0.7, "xyxy": [1, 1, 5, 5]}],
    }
    metrics = env["v2_detector_metrics"]([row])
    assert (metrics["bottle"]["TP"], metrics["bottle"]["FP"], metrics["bottle"]["FN"]) == (1, 1, 0)
    assert metrics["bottle"]["precision_denominator"] == 2
    assert metrics["bottle"]["recall_denominator"] == 1
    assert (metrics["cap"]["TP"], metrics["cap"]["FP"], metrics["cap"]["FN"]) == (0, 0, 1)
    assert metrics["label"]["ground_truth_denominator"] == 0
    assert metrics["label"]["prediction_denominator"] == 1


def test_reviewed_eval_manifest_binds_pair_group_and_source_time(tmp_path):
    env = v2_functions()
    root = tmp_path / "drive"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"authorized source video")
    record = {
        "split": "val", "image_sha256": "i" * 64, "label_sha256": "l" * 64,
        "source_group_id": "session-1", "source_video_drive_relative": "video.mp4",
        "source_video_sha256": "v" * 64, "source_frame_index": 12, "source_time_seconds": 0.4,
    }
    item = {
        "id": "val-1", "split": "val", "image": "images/val/a.jpg",
        **record, "case_tags": sorted(env["V2_EVAL_REQUIRED_CASE_TAGS"]),
    }
    manifest = {"candidate_zip_sha256": "c" * 64, "review_complete": True, "records": [item]}
    validated = env["v2_validate_eval_manifest"](
        manifest, "c" * 64, {item["image"]: record}, "val", root,
    )
    assert validated == [item]
    changed = dict(item, source_time_seconds=0.9)
    with pytest.raises(RuntimeError, match="provenance differs"):
        env["v2_validate_eval_manifest"](
            dict(manifest, records=[changed]), "c" * 64, {item["image"]: record}, "val", root,
        )


def test_paired_playback_decodes_same_frames_and_keeps_source_time(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 4, (32, 32))
    if not writer.isOpened():
        pytest.skip("MJPG decoder/writer is unavailable")
    for index in range(8):
        writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
    writer.release()

    class Model:
        def predict(self, frame, **_kwargs):
            return [SimpleNamespace(plot=lambda: frame.copy())]

    env = v2_functions()
    result = env["v2_render_paired_source_video"](
        source, env["v2_sha256"](source), Model(), Model(), tmp_path / "paired.mp4",
    )
    assert result["frames"] == 8
    assert result["fps"] == 4
    assert [row["frame"] for row in result["timestamps"]] == list(range(8))
    assert [row["source_time_seconds"] for row in result["timestamps"]] == [index / 4 for index in range(8)]
    paired = cv2.VideoCapture(str(tmp_path / "paired.mp4"))
    assert int(paired.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    assert paired.get(cv2.CAP_PROP_FPS) == 4
    ok, frame = paired.read()
    assert ok and frame.shape[1] == 64
    paired.release()


def test_batch_one_timing_uses_real_decoder_and_reports_sample_coverage(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from contextlib import nullcontext

    video = tmp_path / "timing.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (32, 32))
    if not writer.isOpened():
        pytest.skip("MJPG decoder/writer is unavailable")
    for index in range(500):
        writer.write(np.full((32, 32, 3), (index * 17) % 255, dtype=np.uint8))
    writer.release()
    syncs = []

    class Cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def synchronize(_device): syncs.append("sync")
        @staticmethod
        def get_device_name(_device): return "fixture-cuda"

    torch = ModuleType("torch")
    torch.cuda = Cuda
    torch.version = SimpleNamespace(cuda="fixture")
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Predictor:
        def preprocess(self, frames): return frames
        def inference(self, image): return image
        def postprocess(self, raw, _image, _original): return [raw]

    class Model:
        def __init__(self):
            self.predictor = Predictor()
            self.model = SimpleNamespace(stride=[32])
            self.warmups = 0
        def predict(self, _frame, **kwargs):
            assert kwargs["device"] == 0 and kwargs["imgsz"] == 640
            self.warmups += 1

    models = {"v1": Model(), "v2": Model()}
    env = v2_functions()
    report, rows = env["v2_benchmark_real_decoder"](
        video, env["v2_sha256"](video), models, device=0, stride=4, warmup=10, min_samples=120,
    )
    assert models["v1"].warmups == models["v2"].warmups == 10
    assert report["decoded_frames"] == 500
    assert report["sampled_frames"] == len(rows) == 124
    assert report["warmup_source_frames_excluded"] == 1
    assert report["sampling_dropped_frames"] == 375
    assert report["compute_dropped_frames"] is None
    assert rows[0]["source_frame_index"] == 4
    assert rows[-1]["source_frame_index"] == 496
    assert rows[0]["source_time_seconds"] == 1.0
    assert report["batch"] == 1 and report["device"] == "cuda:0"
    assert report["models"]["v1"]["samples"] == report["models"]["v2"]["samples"] == 124
    assert syncs


def test_reviewed_decision_artifact_does_not_promote_historical_proposals():
    review_dir = NOTEBOOK.parents[1] / "tests/fixtures/notebook-reviews"
    decisions = json.loads((review_dir / "candidate-v2-reviewed-decisions-2026-11-05-v1.json").read_text())
    proposals = json.loads((review_dir / "candidate-v2-corrections-2026-11-05-v1.json").read_text())
    assert decisions["candidate_authorized"] is False and decisions["training_ready"] is False
    assert decisions["historical_proposal_artifact_role"] == "comparison_only_never_promoted_automatically"
    rows = {row["id"]: row for row in decisions["decisions"]}
    assert {"train-005", "train-016", "train-018", "train-020", "train-027", "train-044", "train-046"} <= set(rows)
    assert rows["train-020"]["previously_unresolved"] is True
    assert rows["train-020"]["replacement_yolo"].startswith("0 ")
    assert rows["train-044"]["decision"] == rows["train-046"]["decision"] == "exclude"
    assert proposals.get("candidate_authorized", False) is False
    assert proposals["training_ready"] is False and proposals.get("status") != "approved"


@pytest.mark.parametrize("problem", [None, "wrong_archive", "wrong_id", "changed_image", "wrong_review_dataset"])
def test_repair_pack_binds_original_bytes_and_keeps_edits_pending(tmp_path, problem):
    archive = tmp_path / "candidate.zip"
    records = []
    originals = {}
    with zipfile.ZipFile(archive, "w") as z:
        for i in range(2):
            image_path, label_path = f"images/train/source-{i}.jpg", f"labels/train/source-{i}.txt"
            stream = io.BytesIO()
            Image.new("RGB", (100 + i * 10, 80), (100 + i * 30, 40, 30)).save(stream, format="JPEG")
            image = stream.getvalue()
            label = b"0 .5 .5 .6 .8\n"
            z.writestr(image_path, image)
            z.writestr(label_path, label)
            originals[image_path] = image
            originals[label_path] = label
            records.append(dict(split="train", image=image_path, label=label_path,
                                image_sha256="0" * 64 if problem == "changed_image" else hashlib.sha256(image).hexdigest(),
                                label_sha256=hashlib.sha256(label).hexdigest()))
        z.writestr("retained-checksums.json", json.dumps(records))
    ordered = sorted(records, key=lambda r: hashlib.sha256(("label-audit-v1:" + r["image"]).encode()).hexdigest())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    audit = dict(dataset_sha256=digest, samples=[dict(id=f"train-{i + 1:03d}", **r) for i, r in enumerate(ordered)])
    review = dict(dataset_sha256=digest, audit_id="fixture", findings=[dict(id="train-001", reason="Verify omitted cap")])
    if problem == "wrong_id":
        audit["samples"][0].update(ordered[1])
    if problem == "wrong_review_dataset":
        review["dataset_sha256"] = "1" * 64
    output = tmp_path / "repair"
    before = archive.read_bytes()
    env = cell_functions()
    if problem:
        with pytest.raises(RuntimeError):
            env["export_v2_repair_sources"](archive, audit, review, output,
                                              "0" * 64 if problem == "wrong_archive" else digest)
        assert not output.exists()
    else:
        plan = env["export_v2_repair_sources"](archive, audit, review, output, digest)
        assert plan["status"] == "REVIEW_PACK_ONLY" and plan["training_ready"] is False
        assert plan["source_modified"] is False and plan["candidate_v2_created"] is False
        entry = plan["entries"][0]
        assert entry["decision"] == "pending" and entry["replacement_yolo"] is None
        assert (output / entry["source_file"]).read_bytes() == originals[entry["image"]]
        assert (output / entry["original_label_file"]).read_bytes() == originals[entry["label"]]
        assert entry["original_yolo"] == originals[entry["label"]].decode()
        with Image.open(output / entry["source_file"]) as source:
            assert source.size == (entry["width"], entry["height"])
        with Image.open(output / entry["current_overlay"]) as preview:
            assert preview.size == (entry["width"], entry["height"])
        assert json.loads((output / "v2-repair-review.json").read_text(encoding="utf-8"))["training_ready"] is False
    assert archive.read_bytes() == before


@pytest.mark.parametrize("platform,cuda,expected_error", [("win32", True, True), ("linux", False, True), ("linux", True, False)])
def test_speed_guard_requires_remote_cuda_without_fallback(monkeypatch, platform, cuda, expected_error):
    import pathlib
    from types import SimpleNamespace

    env = cell_functions()
    env["SPEED_DEVICE"] = 0
    google, colab, torch = ModuleType("google"), ModuleType("google.colab"), ModuleType("torch")
    google.__path__ = []
    google.colab = colab
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        patch.setattr(pathlib, "Path", lambda value: SimpleNamespace(is_dir=lambda: True))
        patch.setitem(sys.modules, "google", google)
        patch.setitem(sys.modules, "google.colab", colab)
        patch.setitem(sys.modules, "torch", torch)
        if expected_error:
            with pytest.raises(RuntimeError, match="Colab"):
                env["require_speed_gpu"]()
        else:
            assert env["require_speed_gpu"]() is torch


@pytest.mark.parametrize("too_many", [False, True])
def test_timing_reads_real_video_and_synchronizes_each_prediction(tmp_path, too_many):
    import cv2
    import numpy as np

    video = tmp_path / "timing-fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for i in range(320):
        writer.write(np.full((32, 32, 3), i % 255, dtype=np.uint8))
    writer.release()
    calls, order = [], []

    class Model:
        def predict(self, frame, **kwargs):
            calls.append((float(frame.mean()), kwargs))
            order.append("predict")
            return []

    def synchronize():
        order.append("sync")

    fn = cell_functions()["benchmark_prediction_video"]
    if too_many:
        with pytest.raises(RuntimeError, match="frame"):
            fn(Model(), video, 0, synchronize, samples=200, warmup=2)
        assert not calls
        return
    report, rows = fn(Model(), video, 0, synchronize, samples=8, warmup=2)
    assert order == ["predict", "predict", "sync"] + ["sync", "predict", "sync"] * 8
    assert len(calls) == 10 and len(rows) == 8
    assert all(c[1] == dict(device=0, imgsz=640, conf=0.25, verbose=False) for c in calls)
    eligible = [i for i in range(320) if 11 <= i / 10 < 22 or 28 <= i / 10 < 31.5]
    indices = [eligible[i * (len(eligible) - 1) // 7] for i in range(8)]
    assert [r["frame"] for r in rows] == indices
    assert [c[0] for c in calls[2:]] == pytest.approx([i % 255 for i in indices], abs=2)
    assert [r["source_seconds"] for r in rows] == [i / 10 for i in indices]
    assert report["source_fps"] == 10 and report["source_frame_budget_ms"] == 100
    assert report["predict_mean_ms"] > 0
    assert report["production_realtime_proven"] is False and report["accuracy_measured"] is False
    assert "tracking" in report["excludes"] and "render" in report["excludes"]
