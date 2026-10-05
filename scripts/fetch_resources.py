"""Download the upstream specifications and sources used by the harness.

``--only jit-cache`` fetches just the FlashInfer sm80 JIT-cache wheel and
extracts its embedded sm_80 cubins (``cuobjdump -xelf``) into
``resources/flashinfer-jit-cache-sm80/<module>/``: the source of the sm_86
workloads of the fmha, batched_gemm and deep_gemm packages (sm_80 SASS runs
unchanged on sm_86).

``--only flashinfer-v069`` fetches the FlashInfer v0.6.9 source tree and
``--only trtllm-gen`` the trtllm-gen export headers and meta-info of the
artifact paths that release pins: ``cubins/`` was taken from the
``flashinfer_cubin`` 0.6.8/0.6.9 wheel (identical cubins; 10,560 of them), whose
artifact paths and checksums v0.6.8 and v0.6.9 share. Every cubin of
``cubins/batched_gemm`` and ``cubins/gemm`` is listed with its SHA256 in the
pinned ``checksums.txt``.

``--only sources`` fetches the pinned source trees the compilers build
against: FlashInfer ``FLASHINFER_REVISION`` (dsa_attention, fp8_gemm, gdn,
gqa_decode, nvfp4_*, rmsnorm), CUTLASS ``CUTLASS_REVISION``, FlashInfer
v0.2.10 (moe) and a DeepGEMM git checkout (dsa_indexer verifies its revision
with git). Present trees are kept; the compilers check every file they use.

``--only compile`` fetches everything ``compile_kernels.py`` needs (sources,
jit-cache, flashinfer-v069, trtllm-gen) without the definitions; the moe
compiler downloads its own GEMM cubins. Without ``--only`` everything is
fetched.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
CUTLASS_REVISION = "b46b16d003484063bca4ed365e44095c4c6ed633"
# FlashInfer v0.2.10, the moe package's kernel sources (impls/moe/compiler.py).
FLASHINFER_MOE_REVISION = "7c79b41b8efd512eb40ec9bd56c9e6cda328d12e"
# DeepGEMM, the dsa_indexer package's kernel sources (impls/dsa_indexer).
DEEPGEMM_REPOSITORY = "https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_REVISION = "78b69000794d0937b47ae3387eff7663410264d1"
DEEPGEMM_DIR = ROOT / "resources" / "DeepGEMM"
# resources/ directory -> (GitHub repository, revision) of each source tarball.
SOURCE_TREES = {
    f"flashinfer-{FLASHINFER_REVISION}": (
        "flashinfer-ai/flashinfer",
        FLASHINFER_REVISION,
    ),
    f"cutlass-{CUTLASS_REVISION}": ("NVIDIA/cutlass", CUTLASS_REVISION),
    "flashinfer-moe-v0210": ("flashinfer-ai/flashinfer", FLASHINFER_MOE_REVISION),
}
MANIFEST = {}
# FlashInfer 0.7.0 (the release of FLASHINFER_REVISION) AOT JIT-cache provider
# for sm80; flashinfer.ai/whl/cu130 lists it with this digest.
JIT_CACHE_WHEEL = (
    "flashinfer_jit_cache_sm80-0.7.0+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
)
JIT_CACHE_URL = (
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0/"
    + JIT_CACHE_WHEEL
)
JIT_CACHE_SHA256 = "c4c1ba2a6ce557d32eb4b735796af283380d0d1425e73a5e392604885e8fda8b"
JIT_CACHE_DIR = ROOT / "resources" / "flashinfer-jit-cache-sm80"
# FlashInfer v0.6.9: the release whose artifact paths hold the cubins in cubins/.
FLASHINFER_V069_REVISION = "a1aa676196f798435248d9ea205c67674476f473"
FLASHINFER_V069_DIR = ROOT / "resources" / f"flashinfer-{FLASHINFER_V069_REVISION}"
# flashinfer/artifacts.py of v0.6.9 (ArtifactPath, CheckSumHash): artifact path
# -> SHA256 of its checksums.txt. Only the non-cubin files (headers, meta-info)
# are fetched; cubins/ already holds the kernels.
TRTLLM_GEN_REPOSITORY = (
    "https://edge.urm.nvidia.com/artifactory/"
    "sw-kernelinferencelibrary-public-generic-local/"
)
TRTLLM_GEN_ARTIFACTS = {
    "39a9d28268f43475a757d5700af135e1e58c9849/batched_gemm-5ee61af-2b9855b": (
        "db06db7f36a2a9395a2041ff6ac016fe664874074413a2ed90797f91ef17e0f6"
    ),
    "31e75d429ff3f710de1251afdd148185f53da44d/gemm-4daf11e-1fddea2": (
        "64b7114a429ea153528dd4d4b0299363d7320964789eb5efaefec66f301523c7"
    ),
    "55bba55929d4093682e32d817bd11ffb0441c749/fmha/trtllm-gen": (
        "f2c0aad1e74391c4267a2f9a20ec819358b59e04588385cffb452ed341500b99"
    ),
    "a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm": (
        "1a2a166839042dbd2a57f48051c82cd1ad032815927c753db269a4ed10d0ffbf"
    ),
}
# Files an artifact path serves without listing them in checksums.txt:
# DeepGEMM's kernel_map.json (cubin -> symbol, SHA256), pinned by v0.6.9's
# KernelMap.KERNEL_MAP_HASH.
TRTLLM_GEN_UNLISTED = {
    "a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm": {
        "kernel_map.json": (
            "f161e031826adb8c4f0d31ddbd2ed77e4909e4e43cdfc9728918162a62fcccfb"
        ),
    },
}
TRTLLM_GEN_DIR = ROOT / "resources" / "trtllm-gen-artifacts"
NAMES = {
    "moe": "moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    "dsa_indexer": "dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
    "dsa_attention": "dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "gdn_decode": "gdn/gdn_decode_qk4_v8_d128_k_last",
    "gdn_prefill": "gdn/gdn_prefill_qk4_v8_d128_k_last",
    "gqa_decode": "gqa_paged/gqa_paged_decode_h32_kv4_d128_ps1",
    "rmsnorm": "rmsnorm/rmsnorm_h7168",
}


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urlopen(url, timeout=60).read()
    path.write_bytes(data)
    MANIFEST[str(path.relative_to(ROOT))] = {
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    print(path.relative_to(ROOT), len(data), flush=True)
    return data


def fetch_jit_cache() -> None:
    """Download the pinned sm80 JIT-cache wheel and extract its cubins.

    Writes ``<module>/<module>.<n>.sm_80.cubin`` (cuobjdump's names) and
    ``index.json``: wheel URL/digest and, per cubin, the shared object it was
    extracted from with that object's SHA256.
    """
    wheel = ROOT / "resources" / JIT_CACHE_WHEEL
    if not wheel.is_file() or _sha256_file(wheel) != JIT_CACHE_SHA256:
        wheel.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(JIT_CACHE_URL, timeout=300) as response, wheel.open("wb") as f:
            shutil.copyfileobj(response, f)
    if _sha256_file(wheel) != JIT_CACHE_SHA256:
        raise RuntimeError(f"{wheel.name}: SHA256 mismatch")
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump (CUDA toolkit) must be on PATH")
    shutil.rmtree(JIT_CACHE_DIR, ignore_errors=True)
    index: dict = {"wheel": JIT_CACHE_URL, "sha256": JIT_CACHE_SHA256, "cubins": {}}
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as tmp:
        for member in sorted(archive.namelist()):
            if not member.endswith(".so") or "/jit_cache/" not in member:
                continue
            module = member.rsplit("/", 2)[-2]
            data = archive.read(member)
            shared = Path(tmp) / Path(member).name
            shared.write_bytes(data)
            out = JIT_CACHE_DIR / module
            out.mkdir(parents=True)
            result = subprocess.run(
                [cuobjdump, "-xelf", "all", str(shared)],
                cwd=out,
                check=False,
                capture_output=True,
                text=True,
            )
            # Host-only libraries (e.g. spdlog) hold no device code.
            if result.returncode and "does not contain" not in result.stderr:
                raise RuntimeError(f"cuobjdump {member}: {result.stderr.strip()}")
            for cubin in sorted(out.glob("*.cubin")):
                index["cubins"][cubin.relative_to(JIT_CACHE_DIR).as_posix()] = {
                    "member": member.split("/jit_cache/", 1)[1],
                    "member_sha256": hashlib.sha256(data).hexdigest(),
                    "sha256": _sha256_file(cubin),
                }
            if not any(out.iterdir()):
                out.rmdir()
    (JIT_CACHE_DIR / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(JIT_CACHE_DIR.relative_to(ROOT), len(index["cubins"]), "cubins", flush=True)


def fetch_tree(
    repository: str, revision: str, target: Path, archive: Path | None = None
) -> Path:
    """Extract the GitHub tarball of ``repository`` at ``revision`` into
    ``target`` (from ``archive`` if given, else downloaded). An existing
    ``target`` (possibly a symlink) is kept; a partial extraction never
    becomes ``target``."""
    if target.exists():
        return target
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        if archive is None:
            archive = Path(tmp) / "source.tar.gz"
            url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
            with urlopen(url, timeout=300) as response, archive.open("wb") as f:
                shutil.copyfileobj(response, f)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        extracted = Path(tmp) / f"{repository.split('/')[1]}-{revision}"
        extracted.rename(target)
    print(target.relative_to(ROOT), flush=True)
    return target


def fetch_flashinfer_v069() -> Path:
    """Download and extract the FlashInfer v0.6.9 source tree (shared by the
    fmha, batched_gemm and deep_gemm packages; their compilers check the
    SHA256 of every file they use)."""
    return fetch_tree(
        "flashinfer-ai/flashinfer", FLASHINFER_V069_REVISION, FLASHINFER_V069_DIR
    )


def fetch_deepgemm() -> Path:
    """Shallow git checkout of DeepGEMM at ``DEEPGEMM_REVISION`` (no
    submodules: dsa_indexer builds against the CUTLASS tree)."""
    target = DEEPGEMM_DIR
    if target.exists():
        return target
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        checkout = Path(tmp) / "DeepGEMM"
        for command in (
            ["init", "--quiet", str(checkout)],
            ["-C", str(checkout), "remote", "add", "origin", DEEPGEMM_REPOSITORY],
            ["-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin"]
            + [DEEPGEMM_REVISION],
            ["-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(["git", *command], check=True)
        checkout.rename(target)
    print(target.relative_to(ROOT), flush=True)
    return target


def fetch_sources(archives: dict[str, Path] | None = None) -> None:
    """Every pinned source tree of ``SOURCE_TREES`` plus DeepGEMM;
    ``archives`` maps a tree to an already downloaded tarball."""
    for name, (repository, revision) in SOURCE_TREES.items():
        archive = (archives or {}).get(name)
        fetch_tree(repository, revision, ROOT / "resources" / name, archive)
    fetch_deepgemm()


def fetch_compile_inputs() -> None:
    """Everything ``compile_kernels.py`` reads from ``resources/``."""
    fetch_sources()
    fetch_jit_cache()
    fetch_flashinfer_v069()
    fetch_trtllm_gen_artifacts()


def fetch_trtllm_gen_artifacts() -> None:
    """Download checksums.txt and every non-cubin file it lists (export
    headers, flashinferMetaInfo.h) of each TRTLLM_GEN_ARTIFACTS path into
    ``resources/trtllm-gen-artifacts/<artifact path>/``, verifying
    checksums.txt against v0.6.9's CheckSumHash and each file against it."""
    for path, digest in TRTLLM_GEN_ARTIFACTS.items():
        base = TRTLLM_GEN_REPOSITORY + path + "/"
        out = TRTLLM_GEN_DIR / path
        listing = urlopen(base + "checksums.txt", timeout=60).read()
        if hashlib.sha256(listing).hexdigest() != digest:
            raise RuntimeError(f"{path}/checksums.txt: SHA256 mismatch")
        out.mkdir(parents=True, exist_ok=True)
        (out / "checksums.txt").write_bytes(listing)
        count = 0
        files = [line.split()[::-1] for line in listing.decode().splitlines()]
        files += TRTLLM_GEN_UNLISTED.get(path, {}).items()
        for name, expected in files:
            if name.endswith(".cubin"):
                continue
            target = out / name
            if target.is_file() and _sha256_file(target) == expected:
                continue
            data = urlopen(base + name, timeout=300).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"{path}/{name}: SHA256 mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            count += 1
        print(out.relative_to(ROOT), count, "files fetched", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("jit-cache", "flashinfer-v069", "trtllm-gen", "sources", "compile"),
    )
    only = parser.parse_args().only
    if only is not None:
        {
            "jit-cache": fetch_jit_cache,
            "flashinfer-v069": fetch_flashinfer_v069,
            "trtllm-gen": fetch_trtllm_gen_artifacts,
            "sources": fetch_sources,
            "compile": fetch_compile_inputs,
        }[only]()
        return
    for name, path in NAMES.items():
        dataset = (
            "flashinfer-trace"
            if name in ("gqa_decode", "rmsnorm")
            else "mlsys26-contest"
        )
        fetch(
            f"https://huggingface.co/datasets/flashinfer-ai/{dataset}/resolve/main/definitions/{path}.json",
            ROOT / "resources" / f"{name}.json",
        )
    for kind in ("nvfp4_gemm", "nvfp4_dual_gemm", "nvfp4_group_gemm"):
        for filename in ("reference.py", "task.yml"):
            fetch(
                f"https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/nvidia/{kind}/{filename}",
                ROOT / "resources" / kind / filename,
            )
    # Preserve the revision used when extracting optimized implementations.
    sha = FLASHINFER_REVISION
    fetch(
        f"https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/{sha}",
        ROOT / "resources" / "flashinfer.tar.gz",
    )
    (ROOT / "resources" / "flashinfer_revision.txt").write_text(sha + "\n")
    fetch(
        f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_REVISION}",
        ROOT / "resources/cutlass.tar.gz",
    )
    for family, name, solution in (
        ("moe", NAMES["moe"].split("/")[1], "flashinfer_wrapper_9sdjf3"),
        (
            "dsa",
            NAMES["dsa_indexer"].split("/")[1],
            "flashinfer_deepgemm_wrapper_2ba145",
        ),
    ):
        short = "moe" if family == "moe" else "dsa_indexer"
        fetch(
            f"https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/resolve/main/solutions/baseline/{family}/{name}/{solution}.json",
            ROOT / "resources" / (short + "_baseline.json"),
        )
    (ROOT / "resources/downloads.json").write_text(
        json.dumps(MANIFEST, indent=2) + "\n"
    )
    fetch_sources(
        {
            f"flashinfer-{sha}": ROOT / "resources" / "flashinfer.tar.gz",
            f"cutlass-{CUTLASS_REVISION}": ROOT / "resources" / "cutlass.tar.gz",
        }
    )
    fetch_jit_cache()
    fetch_flashinfer_v069()
    fetch_trtllm_gen_artifacts()


if __name__ == "__main__":
    main()
