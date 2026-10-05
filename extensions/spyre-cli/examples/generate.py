# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Compile example kernels with torch.compile on the current toolchain.

Usage: python generate.py [--out DIR] [name ...]    (default: every example)

Kernels are not checked in, since they only load on the toolchain that built
them; build them where they will run. Each example is compiled in a fresh
process with its own Inductor cache, so exactly one Spyre kernel directory is
produced, and it is copied to DIR/<name> (default examples/build/<name>).
Examples that compile to more than one kernel are rejected, since spyre launch
runs a single kernel.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "build"


def _compile_one(name, out):
    # Runs in a child process whose TORCHINDUCTOR_CACHE_DIR is already set.
    import torch
    from cases import EXAMPLES, RTOL
    from references import REFERENCES, expected, make_inputs

    inputs = make_inputs(name)
    fn = torch.compile(REFERENCES[name], backend="inductor")
    got = fn(*[t.to("spyre") for t in inputs]).cpu()
    if not torch.allclose(
        got, expected(name, inputs), atol=EXAMPLES[name].atol, rtol=RTOL
    ):
        sys.exit(f"{name}: torch.compile result does not match CPU")

    cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    kernels = glob.glob(os.path.join(cache, "inductor-spyre", "*", "spyreCodeDir"))
    if len(kernels) != 1:
        sys.exit(f"{name}: expected 1 kernel, torch.compile produced {len(kernels)}")

    dest = Path(out) / name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.dirname(kernels[0]), dest)
    print(f"{name}: wrote {dest}")


def generate(names, out=DEFAULT_OUT):
    """Compile ``names`` into ``out``; return the names that failed."""
    os.makedirs(out, exist_ok=True)
    failed = []
    for name in names:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache)
            cmd = [sys.executable, __file__, "--one", name, "--out", str(out)]
            if subprocess.run(cmd, env=env, cwd=HERE, check=False).returncode:
                failed.append(name)
    return failed


def main():
    from cases import EXAMPLES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="name")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        _compile_one(args.one, args.out)
        return
    unknown = sorted(set(args.names) - set(EXAMPLES))
    if unknown:
        parser.error(f"unknown example(s): {' '.join(unknown)}")
    failed = generate(args.names or sorted(EXAMPLES), args.out)
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
