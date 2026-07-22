#!/usr/bin/env python3
"""Enforce checkpoint retention: keep the step-100 baseline plus the 2 most
recent *fully saved* checkpoints. Runs as a polling loop alongside training.

Safety invariants (see repo checkpointer code, cosmos_policy/_src/predict2/checkpointer/dcp.py):
- latest_checkpoint.txt is only updated by the trainer AFTER dcp.save() fully completes.
- We only ever delete an iter_* dir whose iteration is strictly less than the
  iteration currently named in latest_checkpoint.txt, so we never touch a
  checkpoint that might still be mid-write.
- iter_000000100 (the validated baseline) is never deleted.
"""

import re
import shutil
import sys
import time
from pathlib import Path

BASELINE_ITER = 100
KEEP_RECENT = 1  # in addition to the baseline
POLL_SECONDS = 60

ITER_RE = re.compile(r"^iter_(\d{9})$")


def iter_num(name: str):
    m = ITER_RE.match(name)
    return int(m.group(1)) if m else None


def prune_once(ckpt_dir: Path, log=print):
    latest_file = ckpt_dir / "latest_checkpoint.txt"
    if not latest_file.exists():
        return
    latest_name = latest_file.read_text().strip()
    latest_iter = iter_num(latest_name)
    if latest_iter is None:
        return

    dirs = []
    for p in ckpt_dir.iterdir():
        if p.is_dir():
            n = iter_num(p.name)
            if n is not None:
                dirs.append((n, p))
    dirs.sort()

    # Only ever consider dirs strictly older than the current latest pointer
    # (i.e. guaranteed fully saved, since the pointer only advances after a
    # successful save).
    older = [(n, p) for n, p in dirs if n < latest_iter]

    # Keep the most recent KEEP_RECENT-1 among "older" (the current latest
    # itself counts as one of the KEEP_RECENT), plus always keep baseline.
    # NB: older[-0:] is the whole list in Python, not empty, so n_extra=0
    # must be handled explicitly rather than relying on negative slicing.
    keep_iters = {BASELINE_ITER, latest_iter}
    n_extra = KEEP_RECENT - 1
    for n, _ in (older[-n_extra:] if n_extra > 0 else []):
        keep_iters.add(n)

    for n, p in dirs:
        # Never touch anything at or beyond the current pointer: it may be a
        # save that is still in progress (this was the bug that deleted
        # iter_000000500 mid-write on 2026-07-22 and crashed training).
        if n >= latest_iter:
            continue
        if n not in keep_iters:
            log(f"[prune] deleting old checkpoint {p} (iter {n}), keeping {sorted(keep_iters)}")
            shutil.rmtree(p, ignore_errors=True)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <checkpoints_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_dir = Path(sys.argv[1])
    print(f"[prune] watching {ckpt_dir}, keeping baseline iter_{BASELINE_ITER:09d} + latest {KEEP_RECENT}", flush=True)
    while True:
        try:
            prune_once(ckpt_dir, log=lambda m: print(m, flush=True))
        except Exception as e:
            print(f"[prune] error: {e}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
