# Colab notebooks

Use the notebook for the task at hand. Saved outputs are evidence from the recorded execution and are not proof beyond what they show. Do not rerun an old cell against a different candidate SHA.

| Notebook | Purpose | Current use |
| --- | --- | --- |
| colab-connection-check.ipynb | Check Colab runtime, GPU visibility and Drive mount | Run when verifying a new Colab runtime; it does not train. |
| colab-dataset-preparation.ipynb | Import, inspect and review the earlier dataset candidate | Contains the older duplicate-review workflow. Check its source SHA before running any cell. |
| colab-blur-filter.ipynb | Build a separate candidate from the user-approved quality decisions | The filtered candidate is already built and verified on Drive; do not rebuild unless intentionally creating a new candidate. |
| colab-training-preparation.ipynb | v1 evaluation/video tools and current staged v2 workflow | Stage 11 is complete; current work is Stage 12 semantic review. Follow docs/training-v2-plan.md and run only the stage required. |

Full datasets, review batches and checkpoints stay on Drive/Colab. Do not download them into the Windows worktree. Colab use does not imply that a GPU is required: review and audit stages may use CPU; the training stage requires a verified GPU runtime.