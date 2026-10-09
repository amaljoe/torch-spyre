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


"""The spyre-cli example ops, and a generator for their kernels.

Each entry in EXAMPLES names an example, the tensors ``spyre launch`` must be
given (inputs, then outputs, in kernel argument order), and how its result is
checked against CPU; REFERENCES holds the op behind it. Importing this module
does not import torch, so tests can be collected without a Spyre device.

Run as a script to compile example kernels on the current toolchain:

    python examples.py [--out DIR] [name ...]    (default: every example)

Kernels are not checked in, since they only load on the toolchain that built
them; build them where they will run. Each example is compiled in a fresh
process with its own Inductor cache, so exactly one Spyre kernel directory is
produced, and it is copied to DIR/<name> (default build/<name>). Examples that
compile to more than one kernel are rejected, since spyre launch runs a single
kernel.
"""

import argparse
import dataclasses
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "build"


@dataclasses.dataclass(frozen=True)
class Example:
    inputs: tuple  # input shapes, in kernel argument order
    output: tuple  # output shape
    atol: float
    dtype: str = "fp16"  # a spyre launch dtype: fp16, fp32 or bf16
    init: str = "randn"  # "randn", or "positive" for ops like div


_M = (512, 1024)

EXAMPLES = {
    # elementwise, binary
    "add": Example((_M, _M), _M, 1e-2),
    "sub": Example((_M, _M), _M, 1e-2),
    "mul": Example((_M, _M), _M, 2e-2),
    "div": Example(((256, 2048), (256, 2048)), (256, 2048), 2e-2, init="positive"),
    # elementwise, unary
    "relu": Example((_M,), _M, 1e-2),
    "exp": Example((_M,), _M, 5e-2),
    "sigmoid": Example((_M,), _M, 1e-2),
    "gelu": Example((_M,), _M, 1e-2),
    # fused multi-op
    "fma": Example((_M, _M, _M), _M, 5e-2),
    "swiglu": Example(((128, 4096), (128, 4096)), (128, 4096), 5e-2),
    # other dtypes and ranks
    "add_fp32": Example((_M, _M), _M, 1e-4, dtype="fp32"),
    "mul_bf16": Example((_M, _M), _M, 5e-2, dtype="bf16"),
    "add_3d": Example(((4, 256, 512), (4, 256, 512)), (4, 256, 512), 1e-2),
    # reductions
    "softmax": Example((_M,), _M, 1e-2),
    "mean_keepdim": Example((_M,), (512, 1), 1e-2),
    # matmul
    "mm": Example((_M, (1024, 256)), (512, 256), 1.0),
    "bmm": Example(((8, 128, 256), (8, 256, 128)), (8, 128, 128), 1.0),
}

RTOL = 1e-2


def tensor_spec(shape, dtype):
    """Format a shape as a ``spyre launch`` tensor argument, e.g. 512x1024@fp16."""
    return "x".join(str(d) for d in shape) + "@" + dtype


def cli_args(name):
    """The ``-i``/``-o`` arguments for launching example ``name``."""
    ex = EXAMPLES[name]
    args = []
    for shape in ex.inputs:
        args += ["-i", tensor_spec(shape, ex.dtype)]
    return args + ["-o", tensor_spec(ex.output, ex.dtype)]


def references():
    """The op behind each example, keyed by name."""
    import torch
    import torch.nn.functional as F

    return {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "relu": torch.relu,
        "exp": torch.exp,
        "sigmoid": torch.sigmoid,
        "gelu": F.gelu,
        "fma": lambda a, b, c: a * b + c,
        "swiglu": lambda a, b: F.silu(a) * b,
        "add_fp32": torch.add,
        "mul_bf16": torch.mul,
        "add_3d": torch.add,
        "softmax": lambda a: torch.softmax(a, dim=-1),
        "mean_keepdim": lambda a: a.mean(dim=-1, keepdim=True),
        "mm": torch.mm,
        "bmm": torch.bmm,
    }


def torch_dtype(name):
    """The torch dtype of example ``name``."""
    import torch

    dtypes = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    return dtypes[EXAMPLES[name].dtype]


def make_inputs(name, seed=0):
    """CPU inputs for example ``name``, in its dtype."""
    import torch

    ex = EXAMPLES[name]
    gen = torch.Generator().manual_seed(seed)
    inputs = []
    for shape in ex.inputs:
        t = torch.randn(shape, generator=gen)
        if ex.init == "positive":
            t = t.abs() + 0.5
        inputs.append(t.to(torch_dtype(name)))
    return inputs


def expected(name, inputs):
    """CPU result for ``inputs``, computed in fp32 and cast to the example dtype."""
    return references()[name](*[t.float() for t in inputs]).to(torch_dtype(name))


def _compile_one(name, out):
    # Runs in a child process whose TORCHINDUCTOR_CACHE_DIR is already set.
    import torch

    inputs = make_inputs(name)
    fn = torch.compile(references()[name], backend="inductor")
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
    parser = argparse.ArgumentParser(description="Compile spyre-cli example kernels.")
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
