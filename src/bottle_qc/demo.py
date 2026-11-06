from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from bottle_qc.pipeline.runner import run_simulation_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic BottleQC simulation and persist overlay/evidence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new or empty output location; defaults to a unique directory under the system temp directory",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="bottleqc-demo-"))
    result = run_simulation_demo(output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Artifacts: {output_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
