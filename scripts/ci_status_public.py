"""CI-статус atomicng без авторизации (репо публичный): последние прогоны + упавшие шаги."""
import json
import urllib.request

BASE = "https://api.github.com/repos/Akseleu-J/atomicng"


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": "atomicng-check"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


runs = get("/actions/runs?per_page=6")
for r in runs.get("workflow_runs", []):
    print(f"{r['name']:<10} {r['conclusion'] or r['status']:<10} {r['head_sha'][:7]}  {r['created_at']}  {r['html_url']}")

print("\n--- jobs of the latest ci run ---")
ci_runs = [r for r in runs.get("workflow_runs", []) if r["name"] == "ci"]
if ci_runs:
    run = ci_runs[0]
    jobs = get(f"/actions/runs/{run['id']}/jobs")
    for j in jobs.get("jobs", []):
        failed = [s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"]
        print(f"  {j['name']}: {j['conclusion']}  failed_steps={failed}")
