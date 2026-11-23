"""Measures the fact-checker on the saved test set (tests/eval_cases): its report card.

Runs the full checker (code checks + LLM claim judge) on every case and prints:
- how many known failures it caught (a failure counts as caught if any flag is raised),
- how many honest lines it wrongly flagged.

This calls the real AI (a few cents per run), so it isn't part of pytest. Run from the project root:
    python scripts/eval_factcheck.py
The test set (cases.json, profile.json, jobs.json) holds real profile data, so it's kept out of git and exists
only locally. Results are saved to tests/eval_cases/last_run.json (also not committed).
"""
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import factcheck  # noqa: E402
from app.services.generation import years_of_experience  # noqa: E402

EVAL = ROOT / "tests" / "eval_cases"


def main() -> None:
    missing = [n for n in ("cases.json", "profile.json", "jobs.json") if not (EVAL / n).exists()]
    if missing:
        sys.exit(f"Missing {', '.join(missing)} in {EVAL}. These files hold real profile data and are kept "
                 "out of git, so they only exist on the machine where they were created.")
    profile = json.loads((EVAL / "profile.json").read_text(encoding="utf-8"))
    jobs = json.loads((EVAL / "jobs.json").read_text(encoding="utf-8"))
    cases = json.loads((EVAL / "cases.json").read_text(encoding="utf-8"))["cases"]
    years = years_of_experience([
        SimpleNamespace(start_date=date.fromisoformat(e["start_date"]),
                        end_date=date.fromisoformat(e["end_date"]) if e.get("end_date") else None)
        for e in profile["experiences"]
    ])

    def run(case):
        job = jobs[case["job"]]
        report = factcheck.check_document(case["text"], profile, job_title=job["job_title"],
                                          job_description=job["job_description"],
                                          motivation_notes=case.get("motivation_notes"), years=years)
        return case, report

    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, cases))
    secs = time.time() - start

    failures = [(c, r) for c, r in results if c["kind"] == "failure"]
    honest = [(c, r) for c, r in results if c["kind"] == "honest"]
    caught = [(c, r) for c, r in failures if r["flags"]]
    false_flags = [(c, r) for c, r in honest if r["flags"]]
    by_code = [c for c, r in failures if any(f["type"] != "claim" for f in r["flags"])]
    by_judge = [c for c, r in failures if any(f["type"] == "claim" for f in r["flags"])]
    errors = [c["id"] for c, r in results if r["status"] != "ok"]

    print(f"\n=== Fact-checker report card ({len(cases)} cases, {secs:.0f}s) ===")
    print(f"Known failures caught: {len(caught)}/{len(failures)}"
          f"   (code checks: {len(by_code)}, LLM judge: {len(by_judge)})")
    print(f"Honest lines wrongly flagged: {len(false_flags)}/{len(honest)}")
    if errors:
        print(f"Cases where the claim check couldn't run: {errors}")

    print("\n--- Failures ---")
    for c, r in failures:
        mark = "CAUGHT" if r["flags"] else "MISSED"
        print(f"[{mark}] {c['id']}: {c['problem']}")
        for f in r["flags"]:
            print(f"     {f['severity']:<11} {f['type']:<15} {f['text'][:70]!r}\n                 -> {f['reason'][:110]}")
    print("\n--- Honest lines ---")
    for c, r in honest:
        mark = "FALSE FLAG" if r["flags"] else "ok"
        print(f"[{mark}] {c['id']}")
        for f in r["flags"]:
            print(f"     {f['severity']:<11} {f['type']:<15} {f['text'][:70]!r}\n                 -> {f['reason'][:110]}")

    (EVAL / "last_run.json").write_text(json.dumps({
        "caught": len(caught), "failures": len(failures), "false_flags": len(false_flags), "honest": len(honest),
        "results": [{"id": c["id"], "kind": c["kind"], "report": r} for c, r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
