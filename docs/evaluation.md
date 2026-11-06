# Evaluation and evidence limits

## Detector evaluation

Detector metrics summarize predictions against reviewed labels on a named split. Report per-class precision, recall, AP50 and AP50–95 with the weights, dataset and framework hashes. Choose settings on validation and keep the final test split independent from tuning.

~~~powershell
python -m bottle_qc.evaluation.detector --manifest models/model-id/manifest.json --models-root models --data data/prepared/yolo-v1/data.yaml --split test --device cpu --imgsz 640 --batch 16 --workers 0 --output artifacts/evaluations/detector-test.json
~~~

The evaluator maps arrays using ap_class_index and keeps a class with no ground-truth instances explicit instead of fabricating a zero score.

## Product-level evaluation

Product-level evaluation compares REAL API events with independently reviewed ground-truth intervals. It reports false pass, false fail, UNINSPECTED, missed products, duplicate events and unmatched events. Ambiguous overlapping intervals require reviewed track mapping. Simulation exports are refused.

~~~powershell
python -m bottle_qc.evaluation.system --ground-truth data/annotations/ground-truth.jsonl --events artifacts/runs/real-run.csv --run-id run-123 --source-id line-a --output artifacts/evaluations/system-run.json
~~~

Detector mAP is not the accuracy of the bottle-inspection decision. Qualitative video review is not a measured error rate. A unit test, generated video, simulated provider or successful inference probe cannot establish product quality, line throughput or pilot readiness.

## Current evidence boundary

The v1 detector has recorded validation/test metrics in [project status](project-status.md). The newer filtered candidate has not been used to train a v2 model. The latest video review found qualitative misses and false detections; it has no frame-level ground truth and does not yield a rate. No live camera, PLC, physical line or pilot evaluation has been completed.