"""Build pinned CRAN C50 in an ignored runtime directory; no R required.

The upstream GPL sources remain next to the process-isolated executable.
Only R allocation/output/registration glue is adapted; learner and probability
code remain unchanged. Sampling via R's RNG is deliberately unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/cran/C50.git"
COMMIT = "02261dcca3a8aac8b540e342b813f1a15a74cc72"
SOURCE_HASH = "e18fc76b0e868e9f0d78fc1ac4443495cc457f509eecabc756835c996067d5d6"


def source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(source.iterdir()):
        if file.suffix in {".c", ".h"}:
            digest.update(file.name.encode() + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Existing CRAN C50 checkout; its source hash must match.")
    args = parser.parse_args()
    runtime = ROOT / ".runtime" / "c50"
    runtime.mkdir(parents=True, exist_ok=True)
    checkout = args.source_dir or runtime / "upstream"
    if args.source_dir is None:
        if not (checkout / ".git").exists():
            subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", URL], check=True)
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", COMMIT], check=True)
    source = checkout / "src"
    if source_hash(source) != SOURCE_HASH:
        raise ValueError("CRAN C50 source checksum mismatch; refusing to compile.")
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("A C compiler is required to build native C5.0.")
    build = runtime / "source"
    build.mkdir(exist_ok=True)
    for file in source.iterdir():
        if file.suffix in {".c", ".h"}:
            shutil.copyfile(file, build / file.name)
    shutil.copyfile(ROOT / "native/c50_compat.h", build / "c50_compat.h")
    (build / "Rmath.h").write_text('#include "c50_compat.h"\n')
    top = (source / "top.c").read_text()
    top = top.split("// Declare the type of each of the arguments", 1)[0]
    top = "\n".join(line for line in top.splitlines() if not line.startswith("#include <R"))
    top = '#include <string.h>\n#include "c50_compat.h"\n' + top
    top = top.replace("static void c50(", "void ugo_c50(").replace("static void predictions(", "void ugo_predictions(")
    (build / "top_standalone.c").write_text(top + "\n")
    shutil.copyfile(ROOT / "native/c50_driver.c", build / "c50_driver.c")
    inputs = [str(file) for file in sorted(build.glob("*.c")) if file.name != "top.c"]
    executable = runtime / "c50_bridge"
    subprocess.run([compiler, "-O2", "-DNDEBUG", "-std=gnu11", "-I", str(build),
                    *inputs, "-lm", "-o", str(executable)], check=True)
    record = {"package": "CRAN C50", "version": "0.2.0", "url": URL, "commit": COMMIT,
              "upstream_source_sha256": SOURCE_HASH,
              "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "driver_sha256": hashlib.sha256((ROOT / "native/c50_driver.c").read_bytes()).hexdigest(),
              "compat_sha256": hashlib.sha256((ROOT / "native/c50_compat.h").read_bytes()).hexdigest(),
              "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
              "license": "GPL-3.0-or-later", "r_sampling_supported": False}
    (runtime / "backend.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Built native C5.0: {executable}")


if __name__ == "__main__":
    main()
