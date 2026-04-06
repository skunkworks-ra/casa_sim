"""
Convert S-cubed sourcelist.1deg.txt to casa_sim YAML source catalog.

Extracts positions and 1.4 GHz flux from the catalog.
All other properties (spectral index, curvature, RM, frac_pol, chi) are randomized.

Column layout of sourcelist (comma-separated):
  0,1   : IDs
  2     : type flag
  3     : RA offset (degrees from phase center)
  4     : DEC offset (degrees from phase center)
  5     : position angle
  6,7   : major, minor axis
  8-11  : log10(S_151MHz), NULL, NULL, NULL
  12-15 : log10(S_610MHz), NULL, NULL, NULL
  16-19 : log10(S_1400MHz), NULL, NULL, NULL
  20-23 : log10(S_4860MHz), NULL, NULL, NULL
  24-27 : log10(S_18000MHz), NULL, NULL, NULL
"""

import numpy as np
import yaml
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------- Config ----------
INPUT_FILE = "sourcelist.1deg.txt"
OUTPUT_FILE = "scubed_sources.yaml"
SEED = 12345

# Phase center — match existing test configs
PHASE_CENTER_RA = "19h59m28.5s"
PHASE_CENTER_DEC = "+40d40m00.0s"
REF_FREQ = "1.4GHz"

# Random parameter ranges
SPIX_RANGE = (-2.0, 0.0)        # spectral index alpha
CURV_RANGE = (-0.5, 0.5)        # curvature beta
RM_RANGE = (-500.0, 500.0)      # rotation measure rad/m^2
FRAC_POL_RANGE = (0.001, 0.20)  # fractional linear polarization
CHI_RANGE = (0.0, 180.0)        # EVPA degrees


def main():
    rng = np.random.default_rng(SEED)

    # Phase center
    pc = SkyCoord(ra=PHASE_CENTER_RA, dec=PHASE_CENTER_DEC, frame="fk5")

    # Read catalog
    with open(INPUT_FILE) as f:
        lines = f.readlines()

    n = len(lines)
    print(f"Parsing {n} sources from {INPUT_FILE}")

    # Pre-generate all random values
    alphas = rng.uniform(*SPIX_RANGE, size=n)
    betas = rng.uniform(*CURV_RANGE, size=n)
    rms = rng.uniform(*RM_RANGE, size=n)
    frac_pols = rng.uniform(*FRAC_POL_RANGE, size=n)
    chis = rng.uniform(*CHI_RANGE, size=n)

    sources = []
    for i, line in enumerate(lines):
        words = line.strip().split(",")

        # Position offsets in degrees
        dra_deg = float(words[3])
        ddec_deg = float(words[4])

        # 1.4 GHz flux from log10
        log_flux = float(words[16])
        flux_jy = 10.0 ** log_flux

        # Convert offset to absolute J2000 coordinate
        # RA offset needs cos(dec) correction
        dec_abs = pc.dec.deg + ddec_deg
        ra_abs = pc.ra.deg + dra_deg / np.cos(np.radians(pc.dec.deg))

        coord = SkyCoord(ra=ra_abs * u.deg, dec=dec_abs * u.deg, frame="fk5")
        direction = (
            f"J2000 {coord.ra.to_string(unit=u.hourangle, sep='hms', precision=3, pad=True)}"
            f" {coord.dec.to_string(unit=u.deg, sep='dms', precision=2, pad=True)}"
        )

        src = {
            "name": f"s{i:05d}",
            "direction": direction,
            "flux": [round(flux_jy, 8)],
            "ref_freq": REF_FREQ,
            "spectral_index": [round(float(alphas[i]), 4),
                               round(float(betas[i]), 4)],
            "shape": "point",
            "rm": round(float(rms[i]), 2),
            "frac_pol": round(float(frac_pols[i]), 4),
            "chi": round(float(chis[i]), 2),
        }
        sources.append(src)

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n} sources parsed")

    # Write YAML
    catalog = {"sources": sources}
    print(f"Writing {len(sources)} sources to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False, width=200)

    # Print summary stats
    fluxes = np.array([s["flux"][0] for s in sources])
    print(f"\nSummary:")
    print(f"  Sources: {len(sources)}")
    print(f"  Flux range: {fluxes.min():.2e} — {fluxes.max():.2e} Jy")
    print(f"  Median flux: {np.median(fluxes):.2e} Jy")
    print(f"  Alpha range: [{SPIX_RANGE[0]}, {SPIX_RANGE[1]}]")
    print(f"  RM range: [{RM_RANGE[0]}, {RM_RANGE[1]}] rad/m^2")


if __name__ == "__main__":
    main()
