"""vLLM calibration v2: isolated venv (vllm+torch inside) – bypasses conflict
Stages: S1 atomicng+export, S2 venv+import vllm,
S3 serve smoke (opt-125m), S4 verdict."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
VENV = WORK / "venv"
VPY = str(VENV / "bin" / "python")
STAGES = {}


def save(name: str, data: dict) -> None:
    (RES / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    STAGES[name] = data
    print(f"[{name}]", json.dumps(data, ensure_ascii=False, default=str)[:500], flush=True)


def sh(argv: list[str], timeout: int = 3600) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()[-1500:]


def main() -> None:
    try:
        import glob

        # S1: Atomicng in Basic Environment → Overlays → Artifacts (regardless of vllm))
        wheels = sorted(glob.glob("/kaggle/input/**/atomicng-*.whl", recursive=True))
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", wheels[-1], "pyyaml"], check=True)
        import atomicng

        model = atomicng.TinyModel(spec=atomicng.TableSpec(name="vllm-calib", n_rows=4096, row_width=16))
        writer = atomicng.HashWriter(model.spec)
        atomicng.write(model, "vLLM calibration v2: isolated venv approach.", writer=writer)
        out = atomicng.export_for_vllm(model.t0, model.memory, str(WORK / "vllm-artifacts"),
                                       model_id="calib-run", overlay_path="inline")
        cfg = json.loads((out / "vllm_config.json").read_text(encoding="utf-8"))
        save("S1_export", {"status": "OK", "artifacts": sorted(p.name for p in out.iterdir()),
                           "delta_rows": cfg["n_delta_rows"], "sha256": cfg["sha256_table"][:16]})

        # S2: isolated venv → vllm (pulls its torch pair, does not conflict with the base))
        code, out = sh([sys.executable, "-m", "venv", str(VENV)], timeout=300)
        if code != 0:
            raise RuntimeError(f"venv: {out}")
        code, out = sh([VPY, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], timeout=600)
        save("S2_venv", {"status": "INSTALLING", "note": "venv established; vllm installation (5–15 min))"})
        code, out = sh([VPY, "-m", "pip", "install", "--quiet", "vllm"], timeout=3600)
        if code != 0:
            raise RuntimeError(f"pip install vllm: {out}")
        save("S2_venv", {"status": "OK", "note": "Vllm is installed in an isolated venv"})

        # S3: serve smoke inside venv
        smoke = (
            "import json\n"
            "import vllm\n"
            "llm = vllm.LLM(model='facebook/opt-125m', max_model_len=128, gpu_memory_utilization=0.6)\n"
            "out = llm.generate(['AtomicNG vLLM v2 calibration:'])[0].outputs[0].text\n"
            "print(json.dumps({'vllm': vllm.__version__, 'gen_len': len(out)}))\n"
        )
        (WORK / "smoke.py").write_text(smoke, encoding="utf-8")
        code, out = sh([VPY, str(WORK / "smoke.py")], timeout=1800)
        if code != 0:
            raise RuntimeError(f"serve smoke: {out}")
        smoke_result = json.loads(out.strip().splitlines()[-1])
        save("S3_serve", {"status": "OK", **smoke_result})

        save("S4_verdict", {"status": "PASS", "verdicts": {k: v.get("status") for k, v in STAGES.items()}})
    except Exception:  # noqa: BLE001
        save("V0_exception", {"status": "EXCEPTION", "traceback": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    main()
