"""
tools/generate_corpus.py -- batch corpus generator.

Runs run_corpus_batch over N fields and (optionally) patchifies them into the
torch-ready memmap stacks consumed by MAD-Clean's PatchCorpusDataset.

Each field: sample spec -> CASA predict -> corrupt -> niter=0 Stokes-I MFS
dirty image (768 guard band) -> model/psf FITS.  With --stacks_dir, all
successful fields are diced into dirty/sky/psf patch stacks.

Usage (from repo root):
    pixi run python tools/generate_corpus.py --n_fields 8 \
        --work_dir corpus_runs/val --stacks_dir corpus_stacks/val

    pixi run python tools/generate_corpus.py --n_fields 100 \
        --work_dir corpus_runs/full --stacks_dir corpus_stacks/full --seed 0

T-RECS catalogs are resolved from $CASA_SIM_TRECS_DIR (default <repo>/data/trecs).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_corpus")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_fields", type=int, required=True,
                   help="Number of fields to simulate.")
    p.add_argument("--work_dir", type=str, required=True,
                   help="Directory for per-field CASA outputs (MS, FITS).")
    p.add_argument("--stacks_dir", type=str, default=None,
                   help="If set, patchify successful fields into memmap stacks here.")
    p.add_argument("--seed", type=int, default=0,
                   help="Master RNG seed for reproducible field sampling.")
    p.add_argument("--patch_size", type=int, default=128,
                   help="Patch edge length in pixels (stacks only).")
    p.add_argument("--configs", type=str, default="A,B,C,D",
                   help="Comma-separated VLA configs to sample from.")
    args = p.parse_args()

    from casa_sim.corpus import run_corpus_batch

    configs = tuple(c.strip().upper() for c in args.configs.split(",") if c.strip())

    log.info("=" * 60)
    log.info("Corpus batch: n_fields=%d seed=%d configs=%s", args.n_fields, args.seed, configs)
    log.info("  work_dir   : %s", args.work_dir)
    log.info("  stacks_dir : %s", args.stacks_dir or "(patchify skipped)")
    log.info("=" * 60)

    t0 = time.perf_counter()
    results = run_corpus_batch(
        args.n_fields,
        args.work_dir,
        seed=args.seed,
        vla_configs=configs,
        patchify_out_dir=args.stacks_dir,
        patch_size=args.patch_size,
    )
    dt = time.perf_counter() - t0

    n_ok = sum(1 for r in results if r.success)
    log.info("=" * 60)
    log.info("Batch complete: %d/%d fields succeeded in %.1f s (%.1f s/field)",
             n_ok, len(results), dt, dt / max(len(results), 1))
    if n_ok < len(results):
        for r in results:
            if not r.success:
                log.warning("  field %d FAILED: %s", r.field_idx, r.error)
    if args.stacks_dir:
        log.info("  stacks written to: %s", args.stacks_dir)
    log.info("=" * 60)

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
