"""Diagnose why scripts/build_candidate_dataset.py does not run. Run from the repo root:

    python scripts/check_setup.py --dataset-root "C:\\path\\to\\Student folder"      (Windows)
    python scripts/check_setup.py --dataset-root "/path/to/Student folder"          (Mac/Linux)

Every line is PASS / FAIL / WARN with the fix. Paste the full output when asking for help.
"""
import argparse
import importlib
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAILS = []


def line(status, msg, fix=""):
    print(f"[{status}] {msg}")
    if status != "PASS" and fix:
        print(f"       -> {fix}")
    if status == "FAIL":
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    a = ap.parse_args()

    print(f"python {platform.python_version()} on {platform.system()} | repo root: {ROOT} | cwd: {Path.cwd()}")
    if Path.cwd().resolve() != ROOT:
        line("WARN", "you are not running from the repo root", f'cd "{ROOT}" first')

    # 1. my files present / old conflicting file removed
    for f in ("src/candidate_data.py", "src/route_labels.py", "scripts/build_candidate_dataset.py"):
        line("PASS" if (ROOT / f).exists() else "FAIL", f"{f} exists",
             "unzip member2_update.zip at the repo root (overwrite)")
    if (ROOT / "src" / "data.py").exists():
        line("FAIL", "old src/data.py still exists (clashes with Member 1's src/data/ folder)",
             "delete it: Mac/Linux `rm src/data.py`, Windows `del src\\data.py`, then `git rm --cached src/data.py` if tracked")
    else:
        line("PASS", "old src/data.py is gone")

    # 2. Member 1's code present
    for f in ("src/data/loader.py", "src/features/candidates.py", "src/features/feature_extraction.py"):
        line("PASS" if (ROOT / f).exists() else "FAIL", f"Member 1 file {f} exists", "git pull (or copy Member 1's src/ folders in)")

    # 3. packages
    for m in ("numpy", "pandas", "sklearn", "joblib", "matplotlib", "pyproj", "shapely", "optuna"):
        try:
            mod = importlib.import_module(m)
            line("PASS", f"{m} {getattr(mod, '__version__', '')}")
        except Exception as e:
            need = {"sklearn": "scikit-learn"}.get(m, m)
            sev = "WARN" if m == "optuna" else "FAIL"
            line(sev, f"{m} not importable ({type(e).__name__})", f"pip install {need}"
                 + ("   (only needed for step 6, tuning)" if m == "optuna" else ""))

    # 4. dataset root
    raw = a.dataset_root
    if any(ch in raw for ch in "<>"):
        line("FAIL", f"dataset root '{raw}' is the placeholder from my instructions, not a real folder",
             "replace it with the REAL path of the folder that contains train/ and student_test/")
        root = None
    else:
        root = Path(raw).expanduser()
        if not root.is_dir():
            line("FAIL", f"dataset root does not exist: {root}",
                 'check spelling; keep quotes around paths with spaces: --dataset-root "C:\\...\\Student folder"')
            root = None
    if root:
        line("PASS", f"dataset root exists: {root}")
        first = None
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        print(f"       folders directly inside the dataset root: {found}")
        test_name = next((n for n in ("student_test", "test") if (root / n).is_dir()), None)
        for split in ("train", test_name):
            if split is None:
                line("FAIL", "no student_test/ or test/ folder found", "pass the right root; build script has --test-split NAME")
                continue
            d = root / split
            trips = sorted(p for p in d.iterdir() if p.is_dir())
            line("PASS" if trips else "FAIL", f"{split}: {len(trips)} trip folders", "wrong folder chosen?")
            if trips and split == "train":
                first = trips[0]
            if split == test_name and trips:
                nt = len(trips)
                line("PASS" if nt == 10 else "WARN", f"{split} has {nt} trips (expected 10 student-test trips)")
                no_route = [t.name for t in trips if not (t / f"{t.name}.route").exists()]
                if no_route:
                    line("FAIL", f"{len(no_route)}/{nt} trips in '{split}' have NO .route file (e.g. {no_route[:3]})",
                         "without the reference route there is no is_correct label, so this folder cannot be "
                         "used for evaluation. It may be evaluator-style data: do not train on it; ask the organisers/Member 1 "
                         "where the 10 labelled student-test trips are")
                else:
                    line("PASS", f"all {nt} '{split}' trips have a .route file")
        if first:
            miss = [e for e in ("track", "nodes", "arcs", "route") if not (first / f"{first.name}.{e}").exists()]
            line("PASS" if not miss else "FAIL", f"first train trip {first.name}: files {'ok' if not miss else 'missing ' + str(miss)}",
                 "expected <trip>/<trip>.track .nodes .arcs .route")

            # 5. end-to-end on one trip
            try:
                from src.data.loader import load_trip
                from src.features.candidates import CandidateGenerator
                from src.features.feature_extraction import extract_features
                from src.candidate_data import apply_aliases
                from src.route_labels import assign_route_labels
                t0 = time.perf_counter()
                trip = load_trip(root, "train", first.name)
                f = apply_aliases(extract_features(trip, CandidateGenerator(100, 8).generate(trip)))
                lab = assign_route_labels(f, trip.route)
                share = lab.groupby("obs_id").is_correct.max().mean()
                line("PASS", f"full pipeline OK on trip {first.name}: {len(lab)} rows, {share:.1%} of observations labelled, "
                             f"{time.perf_counter() - t0:.1f}s")
            except Exception as e:
                import traceback
                traceback.print_exc()
                line("FAIL", f"pipeline failed on one trip: {type(e).__name__}: {e}", "send me the traceback above")

    print("\nRESULT:", "ALL GOOD" if not FAILS else f"{len(FAILS)} problem(s) to fix first")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
