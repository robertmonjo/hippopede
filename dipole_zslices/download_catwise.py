"""Download the CatWISE AGN files used by measure_catwise_cuts.py into ./data/catwise.

Source: CatWISE2020 AGN sample of Secrest et al. 2021 (ApJL 908, L51), public release "A Test of the
Cosmological Principle with Quasars", Zenodo record 8303800 (DOI 10.5281/zenodo.8303800).
The files (~390 MB) are not stored in the repository.  Existing files of the right size are kept.
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "catwise")
RECORD = "https://zenodo.org/records/8303800/files/{}?download=1"
FILES = {
    "catwise_agns_masked_final_w1lt16p5_alpha.fits": 389880000,
    "MASKS_exclude_master_final.fits": 23040,
}


def main():
    os.makedirs(DATA, exist_ok=True)
    for name, size in FILES.items():
        path = os.path.join(DATA, name)
        if os.path.exists(path) and os.path.getsize(path) == size:
            print(f"present: {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(RECORD.format(name), path)
        if os.path.getsize(path) != size:
            raise RuntimeError(f"{name}: unexpected size {os.path.getsize(path)} (expected {size})")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
