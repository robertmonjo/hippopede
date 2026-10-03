"""Download the Pantheon+ distance table and its STAT+SYS covariance into data/pantheon_plus/.

Source: Pantheon+ data release (Scolnic et al. 2022, ApJ 938, 113; Brout et al. 2022,
ApJ 938, 110), https://github.com/PantheonPlusSH0ES/DataRelease, directory
Pantheon+_Data/4_DISTANCES_AND_COVAR.  Files already present with the right size are kept.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "pantheon_plus"
BASE = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
FILES = {"Pantheon+SH0ES.dat": ("Pantheon%2BSH0ES.dat", 579283),
         "Pantheon+SH0ES_STAT+SYS.cov": ("Pantheon%2BSH0ES_STAT%2BSYS.cov", 33284960)}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (remote, size) in FILES.items():
        path = DEST / name
        if path.exists() and path.stat().st_size == size:
            print(f"present: {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE + remote, path)
        if path.stat().st_size != size:
            raise RuntimeError(f"{name}: size {path.stat().st_size}, expected {size}")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
