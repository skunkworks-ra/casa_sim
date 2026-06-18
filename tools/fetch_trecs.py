#!/usr/bin/env python
"""
fetch_trecs.py — Download T-RECS catalog files from CDS VizieR.

T-RECS (Tiered Radio Extragalactic Continuum Simulation, Bonaldi et al. 2019)
is hosted at the CDS at Strasbourg:
  https://cdsarc.cds.unistra.fr/ftp/VII/282/

WARNING
-------
First-time setup will download more than 1 GB of data from a slow academic
FTP mirror. The full medium-tier set (AGN + SFG) is ~1.5 GB compressed.
The CDS server is not fast. Plan for this to take 10-30 minutes on a typical
connection.

The SFG catalogs contain far more sources than AGN at faint flux levels.
Dropping your flux_floor_jy below ~1 mJy will push most of your sky model
into SFG territory, which adds millions of sources and makes the simulation
pipeline substantially slower to set up. Start with AGN only and a conservative
flux floor to validate your setup before going deep.

Usage
-----
    pixi run fetch-trecs                    # AGN medium only (recommended start)
    pixi run fetch-trecs --tier medium      # same
    pixi run fetch-trecs --tier deep        # AGN + SFG deep (smaller, faster)
    pixi run fetch-trecs --pops agn sfg     # include SFG (large download)
    pixi run fetch-trecs --dest /my/data    # custom destination directory
    pixi run fetch-trecs --readme-only      # just fetch the ReadMe
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import urllib.error

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/VII/282/"

# Approximate compressed sizes for user guidance
_FILE_SIZES = {
    'agnsdeep.dat.gz':   '  ~14 MB',
    'agnsmedi.dat.gz':   ' ~298 MB',
    'agnswide.dat.gz':   '~1.7 GB',
    'sfgsdeep.dat.gz':   ' ~330 MB',
    'sfgsmedi.dat.gz':   '~2.5 GB',
}

_TIER_FILES = {
    'deep':  {'agn': 'agnsdeep.dat.gz', 'sfg': 'sfgsdeep.dat.gz'},
    'medium': {'agn': 'agnsmedi.dat.gz', 'sfg': 'sfgsmedi.dat.gz'},
    'wide':   {'agn': 'agnswide.dat.gz', 'sfg': None},   # SFG wide is split into 10 files
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end='', flush=True)
    else:
        mb = downloaded / 1e6
        print(f"\r  {mb:.1f} MB downloaded", end='', flush=True)


def fetch_file(filename: str, dest_dir: str, force: bool = False) -> str:
    """Download a single file from CDS into dest_dir. Returns local path."""
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and not force:
        print(f"  Already exists, skipping: {dest_path}")
        return dest_path

    url = BASE_URL + filename
    size_hint = _FILE_SIZES.get(filename, '')
    print(f"\nDownloading {filename}{size_hint}")
    print(f"  from: {url}")
    print(f"  to:   {dest_path}")

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        print()   # newline after progress bar
        actual_mb = os.path.getsize(dest_path) / 1e6
        print(f"  Done: {actual_mb:.1f} MB written")
    except urllib.error.URLError as exc:
        print()
        print(f"  ERROR downloading {filename}: {exc}", file=sys.stderr)
        if os.path.exists(dest_path):
            os.unlink(dest_path)
        raise

    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download T-RECS catalog files from CDS VizieR.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--tier', choices=['deep', 'medium', 'wide'], default='medium',
        help='Catalog tier to fetch (default: medium)',
    )
    parser.add_argument(
        '--pops', nargs='+', choices=['agn', 'sfg'], default=['agn'],
        help='Populations to fetch (default: agn only)',
    )
    parser.add_argument(
        '--dest', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trecs'),
        help='Destination directory (default: data/trecs/ in repo root)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download even if files already exist',
    )
    parser.add_argument(
        '--readme-only', action='store_true',
        help='Download only the ReadMe file',
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    # Always fetch ReadMe
    print(f"Destination: {args.dest}")
    fetch_file('ReadMe', args.dest, force=args.force)

    if args.readme_only:
        print("\nReadMe downloaded. Run without --readme-only to fetch catalogs.")
        return

    # Warn before SFG
    if 'sfg' in args.pops:
        print()
        print("=" * 70)
        print("WARNING: SFG catalogs requested.")
        print()
        print("  The SFG medium catalog is ~2.5 GB compressed and takes 10-30")
        print("  minutes to download from the CDS server, which is not fast.")
        print("  The SFG population also dominates at faint flux levels -- if you")
        print("  lower flux_floor_jy below ~1 mJy you will have millions of sources")
        print("  and the sky model setup will be slow.")
        print()
        print("  Start with AGN only (--pops agn) to validate your pipeline first.")
        print("=" * 70)
        answer = input("Continue with SFG download? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborting SFG download. Re-run with --pops agn to fetch AGN only.")
            args.pops = [p for p in args.pops if p != 'sfg']
            if not args.pops:
                return

    tier_map = _TIER_FILES[args.tier]
    fetched = []
    for pop in args.pops:
        filename = tier_map.get(pop)
        if filename is None:
            print(f"\nWarning: no {pop.upper()} {args.tier}-tier file defined — skipping.")
            continue
        path = fetch_file(filename, args.dest, force=args.force)
        fetched.append((pop, path))

    print()
    print("=" * 60)
    print("Downloaded files:")
    for pop, path in fetched:
        print(f"  {pop.upper():4s}  {path}")
    print()
    print("Add these paths to your YAML config under sky_model.trecs.catalog_paths:")
    for pop, path in fetched:
        print(f"    {pop}: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
