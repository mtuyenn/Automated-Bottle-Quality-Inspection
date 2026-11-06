# User decisions from a bounded train sample review — 2026-11-06

This was a targeted review of 12 train examples from an earlier Drive review pack. It is not a review of the full candidate. Image and label hashes matched that pack when read; no original labels were edited.

- `train-003476`: the user withdrew the earlier missing-cap/label conclusion because the image is blurry. The object may be a milk bottle; keep the label decision unresolved unless the current candidate bytes provide enough evidence.
- `train-001287`: the user identified the green feature as a tamper-evident ring, not a cap. Do not add a class-1 cap box for that ring.
- `train-002480`: the user identified multiple bottles with a box that merges them. If this exact image/hash is still in a current candidate, review and box each visible bottle separately; no final box coordinates were approved here.
- `train-002150`: the white feature around the neck remained unresolved between a physical label and another object.

The sample also showed that heuristic bottle-without-cap/label flags are not annotation decisions. Some examples simply lack a visible component. These 12 product/sample images did not establish that the dataset represents a production conveyor.

The batch and candidate used for this review were older than the current filtered candidate. Verify each ID and image/label SHA against the current candidate before reusing a decision. The current Stage 12 batch and status are in [project status](../project-status.md) and the [active v2 plan](../training-v2-plan.md).