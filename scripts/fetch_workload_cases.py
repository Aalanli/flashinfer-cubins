"""Snapshot official FlashInfer workload shape inventories, not tensor blobs."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from fetch_resources import NAMES, ROOT


def main():
    out = ROOT / "resources/benchmark_workloads"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, definition in NAMES.items():
        dataset = (
            "flashinfer-trace"
            if name in ("rmsnorm", "gqa_decode")
            else "mlsys26-contest"
        )
        url = f"https://huggingface.co/datasets/flashinfer-ai/{dataset}/resolve/main/workloads/{definition}.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            headers = Path(tmp) / "headers"
            data = subprocess.check_output(
                ["curl", "-fsSL", "--max-time", "60", "-D", str(headers), url]
            )
            revision = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in headers.read_text().splitlines()
                    if line.lower().startswith("x-repo-commit:")
                ),
                None,
            )
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        (out / (name + ".jsonl")).write_bytes(data)
        manifest[name] = {
            "url": url,
            "revision": revision,
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(rows),
            "definition": definition.split("/")[-1],
        }
        print(name, len(rows), "rows", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
