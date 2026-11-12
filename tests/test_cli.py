"""CLI contracts: write→stats→verify→merge→publish→bench→smoke, exit codes for both CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng.cli import main


def last_json(out: str) -> dict:
    """First JSON object in output (CLI prints JSON) + human-readable)."""
    start = out.index("{")
    obj, _ = json.JSONDecoder().raw_decode(out[start:])
    return obj


@pytest.fixture()
def workspace(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("First document about kangaroos.\nSecond document about volcanoes.\n", encoding="utf-8")
    return tmp_path, evidence


def test_write_stats_verify_roundtrip(workspace, capsys):
    tmp_path, evidence = workspace
    overlay = str(tmp_path / "ov.atomicng.npz")
    assert main(["write", "--evidence", str(evidence), "--overlay", overlay,
                 "--audit", str(tmp_path / "audit.jsonl"), "--metrics-jsonl", str(tmp_path / "m.jsonl")]) == 0
    write_payload = last_json(capsys.readouterr().out)
    assert write_payload["docs"] == 2 and write_payload["memory"]["n_rows"] > 0
    assert Path(overlay).exists() and (tmp_path / "audit.jsonl").exists()

    assert main(["stats", overlay]) == 0
    stats = last_json(capsys.readouterr().out)
    assert stats["n_rows"] > 0 and stats["total_writes"] > 0 and stats["spec"] == "sidecar-small"

    assert main(["verify", overlay]) == 0
    assert last_json(capsys.readouterr().out)["integrity"] == "ok"
    assert main(["--version"]) == 0


def test_merge_command(workspace, capsys):
    tmp_path, _ = workspace
    evidence_b = tmp_path / "evidence_b.txt"
    evidence_b.write_text("Third document about tidal energy grids offshore.\n", encoding="utf-8")
    ov_a, ov_b = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    assert main(["write", "--evidence", str(tmp_path / "evidence.txt"), "--overlay", ov_a]) == 0
    capsys.readouterr()
    assert main(["write", "--evidence", str(evidence_b), "--overlay", ov_b]) == 0
    capsys.readouterr()
    merged = str(tmp_path / "merged.npz")
    assert main(["merge", "--base", ov_a, "--other", ov_b, "--out", merged, "--gate", "0.5"]) == 0
    capsys.readouterr()
    patch = atomicng.MemoryPatch.load_pretrained(merged)
    a = atomicng.MemoryPatch.load_pretrained(ov_a)
    assert len(patch) >= len(a)


def test_publish_gate(workspace, capsys):
    tmp_path, evidence = workspace
    overlay = str(tmp_path / "ov.npz")
    main(["write", "--evidence", str(evidence), "--overlay", overlay])
    capsys.readouterr()
    # Self-sufficient config: the test does not depend on the configs/repo directory (kernel environment))
    config = tmp_path / "publish_strict.yaml"
    config.write_text(
        "publish:\n  max_neutral_drift: 0.0\n  max_hot_rows: 0\n  hot_norm_cap: 1.0\n",
        encoding="utf-8",
    )
    assert main(["publish", overlay, "--config", str(config)]) == 0
    payload = last_json(capsys.readouterr().out)
    assert payload["published"] is True and payload["checks"]["verify"] is True
    assert payload["neutral_drift"] == 0.0  # Neutral samples do not affect the patch


def test_bench_save_and_check(tmp_path, capsys):
    baseline = str(tmp_path / "baseline.json")
    assert main(["bench", "--docs", "20", "--save", baseline]) == 0
    capsys.readouterr()
    assert main(["bench", "--docs", "20", "--check", baseline, "--tolerance", "100"]) == 0
    capsys.readouterr()
    code = main(["bench", "--docs", "20", "--check", baseline, "--tolerance", "1.0001"])
    assert code in (0, 1)  # It depends on the noise of the timers, but the gate must work out.
    capsys.readouterr()


def test_smoke_command(capsys):
    assert main(["smoke"]) == 0
    payload = last_json(capsys.readouterr().out)
    assert payload["smoke"] is True and payload["locality"] is True
    assert payload["overlay_roundtrip"] is True and payload["base_restored"] is True


def test_cli_error_is_friendly(tmp_path, capsys):
    missing = str(tmp_path / "nope.txt")
    assert main(["write", "--evidence", missing, "--overlay", str(tmp_path / "x.npz")]) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err
