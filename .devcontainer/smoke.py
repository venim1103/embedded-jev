"""Check the research toolchain without downloading or loading a model."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.linalg


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The research container requires Python 3.12.")

    versions = {}
    for executable in ("clang-18", "cmake", "ninja", "git", "shellcheck"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Missing development tool: {executable}")
        result = subprocess.run(
            [executable, "--version"], check=True, capture_output=True, text=True
        )
        versions[executable] = result.stdout.splitlines()[0]

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    factor = scipy.linalg.cho_factor(matrix)
    inverse = scipy.linalg.cho_solve(factor, np.eye(2))
    np.testing.assert_allclose(matrix @ inverse, np.eye(2), atol=1e-12)

    compiler = os.environ.get("CXX", "clang++-18")
    source = """
#include <omp.h>
#include <numeric>
#include <vector>
int main() {
    std::vector<int> values(16, 0);
    #pragma omp parallel for
    for (int index = 0; index < 16; ++index) {
        values[index] = 1;
    }
    return std::accumulate(values.begin(), values.end(), 0) == 16 ? 0 : 1;
}
"""
    with tempfile.TemporaryDirectory(prefix="embedded-jev-smoke-") as directory:
        executable = str(Path(directory) / "openmp-smoke")
        subprocess.run(
            [compiler, "-std=c++17", "-fopenmp", "-x", "c++", "-o", executable, "-"],
            input=source,
            text=True,
            check=True,
        )
        subprocess.run(
            [executable], check=True, env={**os.environ, "OMP_NUM_THREADS": "2"}
        )

    writable_paths = [Path.cwd(), Path(os.environ["HF_HOME"])]
    for directory in writable_paths:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b"embedded-jev environment check\n")
            probe.flush()

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(
        json.dumps(
            {
                "status": "ok",
                "python": platform.python_version(),
                "machine": platform.machine(),
                "tools": versions,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "pytest", "ruff", "uv")
                },
                "checks": ["cholesky", "c++17-openmp", "workspace-write", "cache-write"],
                "model_downloaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()