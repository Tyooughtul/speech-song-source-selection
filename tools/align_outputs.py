#!/usr/bin/env python3
"""Force-align a run file with one shared NeMo CTC model.

This produces the common alignment manifest consumed by the windowed history
analysis: the same aligner is applied to every model's hypotheses, so the
windowed contrasts do not depend on each model's native timestamps.

Example:
    python tools/align_outputs.py \
        --run outputs/primary/temporal_history/runs/whisper.jsonl \
        --output outputs/primary/common_alignment/whisper.jsonl \
        --model-path /path/to/parakeet-ctc-1.1b.nemo \
        --model-revision 20e63a0fed6aedba145b74b826dbd41df0941730
"""

from __future__ import annotations

import argparse

from source_capture.core.alignment import align_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run JSONL to align")
    parser.add_argument("--output", required=True, help="alignment manifest to write")
    parser.add_argument("--model-name", default="nvidia/parakeet-ctc-1.1b")
    parser.add_argument("--model-path", required=True, help="local .nemo artifact")
    parser.add_argument("--model-revision", required=True, help="upstream revision hash")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        align_run(
            args.run,
            args.output,
            {
                "model_name": args.model_name,
                "model_path": args.model_path,
                "resolved_revision": args.model_revision,
                "device": args.device,
                "batch_size": args.batch_size,
                "alignment_batch_size": args.batch_size,
                "word_timestamps": True,
            },
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
