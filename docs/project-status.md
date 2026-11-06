# BottleQC project status

Updated: 2026-11-07. This page describes the current state; detailed evidence remains in the linked reports and original Colab/Drive outputs.

## Application

The agreed product is a Windows desktop application packaged as an .exe. The current code is a local FastAPI service with a Streamlit browser client; it is not yet the desktop application. Still images and sequential uploaded video are implemented. Live camera, PLC/rejector control and deployment are not.

## Model baseline

YOLO11n v1 was trained on a Colab Tesla T4 and saved on Drive at BottleQC/training-runs/bottleqc-yolo11n-640-v1. It completed 92 epochs, with the best checkpoint at epoch 72. Best weights SHA-256: 4839c128b8b13b5785b55db2d0a15d061e424b3abfbca3653f48a1b9a5533030.

| Split | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 92.18% | 87.14% | 92.52% | 62.54% |
| Test | 89.17% | 94.72% | 94.85% | 69.06% |

These are detector metrics on the v1 dataset, not PASS/FAIL accuracy or evidence of performance on a production line. Weights are not stored in the local models directory.

## Current dataset candidate

The user-directed quality filter removed 79 image/label pairs from a separate candidate; the source ZIP was preserved. The current Drive candidate has 3,778 train / 299 val / 201 test pairs and SHA-256 03198fb7279e89fee87c5d91df9b5d0d908fa943766a6981e6f8688871b35e3e. See the [quality-decision report](reports/2026-11-06-blur-user-quality-decision.md).

Stage 11 completed on that exact SHA: 0 structural errors, 0 empty labels, 741 pHash-near groups; 66 groups span splits. Near-hash groups are candidates for review, not confirmed duplicates. The audit is at BottleQC/datasets/candidate-v2-audits/audit-v2-03198fb7279e89fe.

Stage 12 has created the current 72-image batch: semantic-review/batch-4731f0164a1c0f70. Its saved receipt shows 0 reviewed, 0 unresolved and 4,278 unreviewed, with semantic_clean_claim=false. The batch is in the test split. Review it without moving samples between splits. The next action and exact Drive paths are in the [active v2 plan](training-v2-plan.md). No v2 training has started.

## Model behavior on line video

The v1 model has qualitative false detections and missed bottles on the reviewed MV Center line clip. There is no frame-level ground truth for that clip, so the review does not establish an error rate. See the [video review](reports/2026-11-04-mvcenter-video-review.md). Dataset/annotation review and a separately labelled held-out line video are still needed.

## Current evidence

The [training-v2 plan](training-v2-plan.md) is the only active execution plan. The [sample label review](reports/2026-11-06-drive-train-sample-01.md) and [bounded original-pixel review](reports/2026-11-05-original-label-review.md) contain user-relevant annotation observations; their decisions must be checked against the current candidate hashes before reuse. See [evaluation limits](evaluation.md) for what local tests and detector metrics do not prove.