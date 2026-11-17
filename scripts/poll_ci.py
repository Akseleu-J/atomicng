"""Poll CI status for commit 6c8eded."""
import json
import time
import urllib.request

BASE = "https://api.github.com/repos/Akseleu-J/atomicng"
SHA = "6c8eded"


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": "atomicng-check"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


for attempt in range(10):
    try:
        runs = get("/actions/runs?per_page=5")
        ci = [r for r in runs.get("workflow_runs", []) if r["name"] == "ci" and r["head_sha"].startswith(SHA)]
        if ci:
            run = ci[0]
            print(f"attempt {attempt + 1}: ci {run['conclusion'] or run['status']}")
            if run["conclusion"]:
                jobs = get(f"/actions/runs/{run['id']}/jobs")
                for j in jobs.get("jobs", []):
                    failed = [s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"]
                    print(f"  {j['name']}: {j['conclusion']} failed={failed}")
                break
        else:
            print(f"attempt {attempt + 1}: ci run for {SHA} not started yet")
    except Exception as e:
        print(f"attempt {attempt + 1}: error {e!r}")
    time.sleep(60)
