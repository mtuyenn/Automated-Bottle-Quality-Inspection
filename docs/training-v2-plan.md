# Active plan: prepare and train BottleQC v2

Updated: 2026-11-07. Training is not ready to start. Code changes happen in VS Code on Windows; data work and training run in Colab. Do not train on the workstation or move the full dataset to it.

## Current binding

Use only the filtered Drive candidate with SHA-256 03198fb7279e89fee87c5d91df9b5d0d908fa943766a6981e6f8688871b35e3e, counts 3,778 train / 299 val / 201 test. The original source ZIP remains separate. Stage 11 is complete for this SHA, but 741 pHash-near groups remain candidates for review; 66 span splits and are not automatically duplicates.

Stage 12 created batch-4731f0164a1c0f70 with 72 test images. Its receipt records 0 reviewed and 4,278 unreviewed; semantic_clean_claim=false. No v2 candidate or model has been approved.

## Next action: review the current Stage 12 batch

The batch is under:

    MyDrive/BottleQC/datasets/candidate-v2-audits/audit-v2-03198fb7279e89fe/semantic-review/batch-4731f0164a1c0f70/

For each image, compare source pixels, the current overlay and original YOLO text. Preserve the ID and image/label hashes. Record a complete final YOLO label and reviewer/time/policy for a confident decision; record unresolved with a reason when the pixels do not support a reliable decision. Do not use an empty label unless a reviewer confirms a hard negative. Keep the 72 samples in test.

Save the accumulated decision list at:

    MyDrive/BottleQC/datasets/candidate-v2-audits/audit-v2-03198fb7279e89fe/semantic-decisions-reviewed.json

Only after every row in the current batch is reviewed or unresolved, run Stage 12 again to create the next batch. It will repeat the same batch if the decisions file is missing or unchanged. Continue until full-candidate semantic coverage is complete and any unresolved cases are resolved or explicitly blocked by policy. Stage 12 uses Colab CPU; no GPU is needed for review-batch creation.

## Remaining gates before training

1. Complete semantic review bound to the current candidate SHA. Do not infer label completeness from structural checks or class-frequency heuristics.
2. Review cross-split near-hash groups and verify capture-session grouping. Keep unresolved comparisons explicit.
3. Complete rights and capture-group intake for any added media. Use only authorized media and include motion blur, occlusion, reflection, small caps, physical sleeves and hard negatives where supported by source evidence.
4. Complete the required policy/static review and holdout freeze. Build a fresh grouped candidate with lineage and verified checksums; do not edit the source ZIP or the existing v1 dataset.
5. Before training, update the candidate ID/path in the v2 training cell and check the exact candidate SHA, Colab GPU, free runtime disk, Drive write/readback and Ultralytics 8.4.193. Start a new YOLO11n run initialized from the verified v1 best checkpoint; do not resume or overwrite the completed v1 run.
6. Compare v1/v2 on the same reviewed validation data. Keep test untouched until final evaluation. Measure product-level false pass/fail separately from detector mAP; measure full pipeline latency on the intended device before making a real-time claim.

## Video and capture note

The v1 detector qualitatively misses bottles and creates false detections on the MV Center line clip; the clip has no ground truth, so no error rate is available. Faster belt motion also requires short exposure and sufficient lighting to preserve visible detail. Blur augmentation cannot recover details missing from the source image. See the [video review](reports/2026-11-04-mvcenter-video-review.md).