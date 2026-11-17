"""vLLM calibration v3: uv-venv (without ensurepip) → vllm → serve smoke → verdict."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
VENV = WORK / "v.env"
VPY = str(VENV / "bin" / "python")


def save(name: str, data: dict) -> None:
    (RES / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[{name}]", json.dumps(data, ensure_ascii=False, default=str)[:400], flush=True)


def sh(argv: list[str], timeout: int = 3600) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()[-1200:]


def main() -> None:
    try:
        import glob

        wheels = sorted(glob.glob("/kaggle/input/**/atomicng-*.whl", recursive=True))
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", wheels[-1], "pyyaml"], check=True)
        import atomicng

        model = atomicng.TinyModel(spec=atomicng.TableSpec(name="vllm-calib", n_rows=4096, row_width=16))
        writer = atomicng.HashWriter(model.spec)
        atomicng.write(model, "vLLM calibration v3: uv isolated venv.", writer=writer)
        out = atomicng.export_for_vllm(model.t0, model.memory, str(WORK / "vllm-artifacts"),
                                       model_id="calib-run", overlay_path="inline")
        cfg = json.loads((out / "vllm_config.json").read_text(encoding="utf-8"))
        save("S1_export", {"status": "OK", "artifacts": sorted(p.name for p in out.iterdir()),
                           "delta_rows": cfg["n_delta_rows"]})

        code, out = sh([sys.executable, "-m", "pip", "install", "--quiet", "uv"], timeout=900)
        if code != 0:
            raise RuntimeError(f"pip install uv: {out}")
        code, out = sh([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable], timeout=600)
        if code != 0:
            raise RuntimeError(f"uv venv: {out}")
        save("S2_uv_venv", {"status": "OK", "venv": VENV})
        code, out = sh([sys.executable, "-m", "uv", "pip", "install", "--python", VPY, "vllm"], timeout=3600)
        if code != 0:
            raise RuntimeError(f"uv pip install vllm: {out}")
        save("S3_uv_install", {"status": "OK"})

        smoke = (
            "import json\n"
            "import vllm\n"
            "llm = vllm.LLM(model='facebook/opt-125m', max_model_len=128, gpu_memory_utilization=0.6)\n"
            "out = llm.generate(['AtomicNG vLLM v3 calibration:'])[0].outputs[0].text\n"
            "print(json.dumps({'vllm': vllm.__version__, 'gen_len': len(out)}))\n"
        )
        (WORK / "smoke.py").write_text(smoke, encoding="utf-8")
        code, out = sh([VPY, str(WORK / "smoke.py")], timeout=1800)
        if code != 0:
            raise RuntimeError(f"serve smoke: {out}")
        smoke_result = None
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    smoke_result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if smoke_result is None:
            raise RuntimeError(f"Smoke json not found in the output: {out[-300:]}")
        save("S4_serve", {"status": "OK", **smoke_result})
        save("S5_verdict", {"status": "PASS"})
    except Exception:  # noqa: BLE001
        save("V0_exception", {"status": "EXCEPTION", "traceback": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    main()
