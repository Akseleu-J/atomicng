"""vLLM calibration on Kaggle GPU (Linux): installation of vllm, artifacts AtomicNG, serve smoke.

Honest coverage: Pipeline export_for_vllm + running the engine on the same machine.
inside the graph engine - Track B/upstream (docs/vllm.md path 3); the environment is proved here.
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
STAGES = {}


def save(name: str, data: dict) -> None:
    (RES / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    STAGES[name] = data
    print(f"[{name}]", json.dumps(data, ensure_ascii=False, default=str)[:500], flush=True)


def pip(args: list[str], timeout: int = 2400) -> int:
    return subprocess.run([sys.executable, "-m", "pip", "install", *args],
                          capture_output=True, text=True, timeout=timeout).returncode


def main() -> None:
    try:
        import glob
        import torch

        save("S0_probe", {"status": "OK", "python": sys.version.split()[0], "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"})

        wheels = sorted(glob.glob("/kaggle/input/**/atomicng-*.whl", recursive=True))
        code = pip(["--quiet", wheels[-1]]) if wheels else 1
        code += pip(["--quiet", "pyyaml"])
        import atomicng

        # S1: overlay → export_for_vllm
        import numpy as np

        model = atomicng.TinyModel(spec=atomicng.TableSpec(name="vllm-calib", n_rows=4096, row_width=16))
        writer = atomicng.HashWriter(model.spec)
        atomicng.write(model, "vLLM calibration overlay document about atomic memory rows.",
                       writer=writer)
        out = atomicng.export_for_vllm(model.t0, model.memory, str(WORK / "vllm-artifacts"),
                                       model_id="calib-run", overlay_path="inline")
        cfg = json.loads((out / "vllm_config.json").read_text(encoding="utf-8"))
        merged = np.load(out / "ngram_table_merged.npz")["table"]
        save("S1_export", {"status": "OK", "artifacts": sorted(p.name for p in out.iterdir()),
                           "merged_shape": list(merged.shape), "sha256": cfg["sha256_table"][:16],
                           "delta_rows": cfg["n_delta_rows"]})

        # S2: install vllm (heavy) and run the engine on the same machine
        save("S2_vllm_install", {"status": "RUNNING"})
        code = pip(["--quiet", "vllm"], timeout=3600)
        if code != 0:
            save("S2_vllm_install", {"status": "FAILED", "pip_exit": code})
        else:
            import vllm

            t0 = time.monotonic()
            llm = vllm.LLM(model="facebook/opt-125m", max_model_len=256, gpu_memory_utilization=0.6)
            out_txt = llm.generate(["AtomicNG vLLM bridge calibration:"])[0].outputs[0].text
            save("S3_vllm_serve", {
                "status": "OK", "vllm": vllm.__version__,
                "engine_init_min": round((time.monotonic() - t0) / 60, 1),
                "sample_len": len(out_txt),
                "note": "The engine runs on a Linux GPU next to AtomicNG artifacts; "
                        "graph-based merged tables - Track B (docs/vllm.md path 3)",
            })
    except Exception:  # noqa: BLE001
        save("V0_exception", {"status": "EXCEPTION", "traceback": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    main()
