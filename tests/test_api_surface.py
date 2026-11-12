"""Golden Public API Test: The Surface Frozen Before v1.0.

Change this list = conscious decision (update the golden list and CHANGELOG)
A random change is a release error.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng

GOLDEN_API = [
    "__version__",
    "AuditLog",
    "AtomicNGError",
    "BackendError",
    "Block",
    "BudgetExceeded",
    "ByteTokenizer",
    "D1Rule",
    "Event",
    "GuardViolation",
    "HashWriter",
    "JsonlExporter",
    "MemoryPatch",
    "MetricsRegistry",
    "OverlayError",
    "PublishConfig",
    "PublishRejected",
    "SIDECAR_SMALL",
    "SpecError",
    "TableSpec",
    "TinyModel",
    "UndoJournal",
    "VerifyReport",
    "WritePolicies",
    "WriteReport",
    "acquisition",
    "attach_memory",
    "build_blocks",
    "build_blocks_from_texts",
    "build_report",
    "chain",
    "check_regression",
    "clear_memory",
    "distraction",
    "export_for_vllm",
    "export_ple",
    "generate_keypair",
    "load_metrics_jsonl",
    "load_ple",
    "merged_table",
    "publish_check",
    "revert_last",
    "run_bench",
    "run_suite",
    "save_baseline",
    "sign_file",
    "specs",
    "vllm_general_plugins",
    "verify_deletion",
    "verify_file",
    "write",
    "write_report",
]


def test_public_api_frozen():
    public = sorted(n for n in atomicng.__all__)
    golden = sorted(GOLDEN_API)
    added = set(public) - set(golden)
    removed = set(golden) - set(public)
    assert not added and not removed, (
        "Public API changed before v1.0! "
        f"Added: {sorted(added)}; Removed.: {sorted(removed)}. "
        "If this is a conscious change, update Golden._API in the test and the CHANGELOG "
        "Note the API-change; otherwise, return the names.."
    )
    assert sorted(atomicng.__all__) == public
    assert public == sorted(public), "__all__ should be sorted"


def test_version_semver_pre1():
    """v1.0.0 – First stable tag: Major must be 1 (updated in major releases))."""
    major = int(atomicng.__version__.split(".")[0])
    assert major == 1, f"expected major 1 (v1.0.0), obtained {atomicng.__version__}"
