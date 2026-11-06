# Data and annotation

## Where data lives

The local data/raw, data/annotations, data/prepared and models directories are placeholders for small authorized development inputs and selected local inference assets. The working BottleQC dataset and training checkpoints are on Google Drive/Colab. Do not copy the full dataset or training frames to Windows.

Keep source data immutable. Make exclusions, annotation changes and split changes in a separately identified candidate, with source hashes and a reason for every action. Keep uncertain quality or label decisions unresolved until there is visual evidence. An exclusion count from an earlier candidate is not a fixed quota.

The current filtered candidate, its 79 user-directed exclusions and its SHA are recorded in [project status](project-status.md) and the [quality-decision report](reports/2026-11-06-blur-user-quality-decision.md).

## Capture groups and splits

Record one manifest row per reviewed image. Paths must be relative to the source root. Preserve source video, capture session and batch identifiers so related captures stay in the same split. A split is not evidence of independent capture sessions unless the source grouping is verified.

The local dataset CLI can inventory, validate, split and prepare an explicitly supplied small dataset. Its defaults are data/raw and data/annotations/labels; it does not download media. Training itself is Colab-only under project rules.

~~~powershell
python -m bottle_qc.training.dataset inventory --root data/raw --annotations data/annotations/labels
python -m bottle_qc.training.dataset validate --root data/raw --annotations data/annotations/labels --manifest data/raw/manifest.jsonl
python -m bottle_qc.training.dataset split --root data/raw --annotations data/annotations/labels --manifest data/raw/manifest.jsonl --output data/prepared/split-manifest.jsonl --seed 42
python -m bottle_qc.training.dataset prepare --root data/raw --annotations data/annotations/labels --manifest data/prepared/split-manifest.jsonl --output data/prepared/yolo-v1
~~~

Preparation writes a new output and preserves the inputs. Invalid paths, unpaired files, undecodable images, malformed labels, duplicate content or split leakage are rejected.

## YOLO annotation policy

Every nonempty label row has five finite values: class_id, center_x, center_y, width, height. Coordinates are normalized to [0,1], width and height are positive, and the box fits inside the image. Classes are fixed: 0=bottle, 1=cap, 2=label.

Review the pixels and the label together. Add one box for each visible instance; do not merge multiple bottles. A partially cropped bottle may be labelled when the visible pixels identify it; do not guess the cropped part. A cap is class 1 only when visible. Class 2 is a separate physical label or sleeve, not lettering printed directly on the bottle. A missing component is not automatically an annotation error. If blur, occlusion or ambiguous appearance prevents a reliable decision, keep it unresolved. An empty label is valid only for a reviewed hard negative with no target.

The training preflight checks class mapping, image decoding, exact image/label pairs, paths, finite normalized coordinates and train/validation leakage. Passing that check does not certify that boxes are semantically correct.