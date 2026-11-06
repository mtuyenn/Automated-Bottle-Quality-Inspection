# User-directed exclusion of low-quality pairs — 2026-11-06

## Decision and result

After reviewing the 37-image comparison set, the user directed that all 37 pairs be excluded from the training candidate because the images were low quality, repetitive/overlapping, illustrative, or otherwise likely to add noise. Combined with 42 prior quality exclusions, the candidate builder removed 79 image/label pairs from the 4,357-pair parent: 71 train, 8 validation and 0 test. The original source ZIP was not modified.

The resulting Drive candidate contains 4,278 pairs: 3,778 train, 299 validation and 201 test. Its SHA-256 is `03198fb7279e89fee87c5d91df9b5d0d908fa943766a6981e6f8688871b35e3e`. The candidate ZIP and its COMPLETE marker were read back from Drive and verified against the source and decision hashes.

The user's exclusion decision records candidate quality, not semantic correctness of every remaining annotation. It does not certify this candidate as training-ready.

## Current artifacts

- Decision manifest: [decisions-full-review-v2-drop-ambiguous.json](https://drive.google.com/file/d/17V_M6EEmGp032ubNyISKTVmgJR5h02zw/view), SHA-256 `62d42a5103387231a4bfadc013a08e68281b2c6079bf68efb691c3f5ebcc1eb0`.
- Candidate folder: [blur-filtered-b17e060caa43d8c3](https://drive.google.com/drive/folders/1RwrDwZPhYl18n2yjTdQKFdxpjRMHsIjD).
- Candidate SHA-256: `03198fb7279e89fee87c5d91df9b5d0d908fa943766a6981e6f8688871b35e3e`.

The Drive manifest is the source of truth for per-ID decisions; this repository does not keep another 6.5 MB copy.

## Follow-up

Validation changed from 307 to 299 images, so compare model versions on the same 299-image split. The test split remains 201. Stage 11/12 audit and semantic-review state are recorded in [project status](../project-status.md). No v2 training has run.