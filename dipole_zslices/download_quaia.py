"""Download the Quaia files used by measure_dipole.py into ./data.

Source: Quaia catalogue (Storey-Fisher et al. 2024, ApJ 964, 69), Zenodo record 10403370.
The files (~270 MB) are not stored in the repository.  Existing files of the right size are kept.
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RECORD = "https://zenodo.org/records/10403370/files/{}?download=1"
FILES = {
    "quaia_G20.0.fits": 99786240,
    "quaia_G20.5.fits": 171020160,
    "selection_function_NSIDE64_G20.0.fits": 400320,
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
