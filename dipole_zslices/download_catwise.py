"""Download the CatWISE AGN files used by measure_catwise_cuts.py into ./data/catwise.

Source: CatWISE2020 AGN sample of Secrest et al. 2021 (ApJL 908, L51), public release "A Test of the
Cosmological Principle with Quasars", Zenodo record 8303800 (DOI 10.5281/zenodo.8303800).
The files (~390 MB) are not stored in the repository.  Existing files of the right size are kept.
With --sdss also the SDSS DR16Q quasar catalogue (Lyke et al. 2020, ApJS 250, 8; 1.7 GB) into ./data/sdss, used by
catwise_redshifts.py.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "catwise")
RECORD = "https://zenodo.org/records/8303800/files/{}?download=1"
FILES = {
    "catwise_agns_masked_final_w1lt16p5_alpha.fits": 389880000,
    "MASKS_exclude_master_final.fits": 23040,
}


SDSS = ("https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/{}", os.path.join(HERE, "data", "sdss"), {"DR16Q_v4.fits": 1718484480})


def main():
    targets = [(RECORD, DATA, FILES)] + ([SDSS] if "--sdss" in sys.argv[1:] else [])
    for url, folder, files in targets:
        fetch(url, folder, files)


def fetch(url, folder, files):
    os.makedirs(folder, exist_ok=True)
    for name, size in files.items():
        path = os.path.join(folder, name)
        if os.path.exists(path) and os.path.getsize(path) == size:
            print(f"present: {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url.format(name), path)
        if os.path.getsize(path) != size:
            raise RuntimeError(f"{name}: unexpected size {os.path.getsize(path)} (expected {size})")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
