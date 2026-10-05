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


"""Registry of the example kernels.

Each entry names an example, the tensors ``spyre launch`` must be
given (inputs, then outputs, in kernel argument order), and how the result is
checked against CPU. Kept free of torch imports so tests can be collected on
machines without a Spyre device. The ops themselves live in references.py.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Example:
    inputs: tuple  # input shapes, in kernel argument order
    output: tuple  # output shape
    atol: float
    dtype: str = "fp16"  # a spyre launch dtype: fp16, fp32 or bf16
    init: str = "randn"  # "randn", or "positive" for ops like sqrt and div


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
