"""Regenerate every certificate and compare with the shipped files.

Usage (from the package root): python3 code/verify_all.py [--skip-numerical]

Each verifier writes into a temporary directory; every output is compared
byte-for-byte with certificates/<name>.json and with its SHA256SUMS entry.
Exit status is nonzero if any file differs, is missing, or a verifier fails.
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXACT = [
    ("verify_tetrahedron.py", "tetrahedron_exact.json"),
    ("verify_dimensions.py", "dimensions_exact.json"),
    ("verify_lean_element.py", "lean_element_exact.json"),
    ("independent_check.py", "independent_check.json"),
]
NUMERICAL = [("stokes_n3_numerical.py", "stokes_n3_numerical.json")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sums():
    sums = {}
    f = ROOT / "SHA256SUMS"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            name = parts[-1].lstrip("*")
            if name.startswith("./"):
                name = name[2:]
            sums[name] = parts[0].lower()
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-numerical", action="store_true",
                    help="do not rerun the floating-point Stokes computation")
    args = ap.parse_args()
    jobs = EXACT + ([] if args.skip_numerical else NUMERICAL)
    sums = read_sums()
    if sums is None:
        print("FAIL  SHA256SUMS not found")
    failed = sums is None
    with tempfile.TemporaryDirectory() as tmp:
        for script, name in jobs:
            out = Path(tmp) / name
            proc = subprocess.run([sys.executable, str(ROOT / "code" / script), "--out", str(out)],
                                  cwd=ROOT, capture_output=True, text=True)
            shipped = ROOT / "certificates" / name
            key = "certificates/" + name
            if proc.returncode != 0 or not out.exists():
                print(f"FAIL  {key}  (verifier exited with status {proc.returncode})")
                sys.stderr.write(proc.stderr[-2000:])
                failed = True
                continue
            same = shipped.exists() and out.read_bytes() == shipped.read_bytes()
            digest = sha256(out)
            sum_ok = sums is not None and sums.get(key) == digest
            status = "PASS" if same and sum_ok else "FAIL"
            failed = failed or status == "FAIL"
            note = []
            if not shipped.exists():
                note.append("shipped file missing")
            elif not same:
                note.append("differs from shipped file")
            if sums is not None and key not in sums:
                note.append("no SHA256SUMS entry")
            elif sums is not None and not sum_ok:
                note.append("SHA256SUMS mismatch")
            print(f"{status}  {key}  {digest}" + (f"  ({'; '.join(note)})" if note else ""))
    if args.skip_numerical:
        print("SKIP  certificates/stokes_n3_numerical.json  (--skip-numerical)")
    print("OVERALL", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
