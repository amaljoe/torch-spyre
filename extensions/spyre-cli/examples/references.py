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


"""The op behind each example in cases.py, and its test inputs."""

import torch
import torch.nn.functional as F
from cases import EXAMPLES

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


REFERENCES = {
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


def make_inputs(name, seed=0):
    """CPU inputs for example ``name``, in its dtype."""
    ex = EXAMPLES[name]
    gen = torch.Generator().manual_seed(seed)
    inputs = []
    for shape in ex.inputs:
        t = torch.randn(shape, generator=gen)
        if ex.init == "positive":
            t = t.abs() + 0.5
        inputs.append(t.to(DTYPES[ex.dtype]))
    return inputs


def expected(name, inputs):
    """CPU result for ``inputs``, computed in fp32 and cast to the example dtype."""
    dtype = DTYPES[EXAMPLES[name].dtype]
    return REFERENCES[name](*[t.float() for t in inputs]).to(dtype)
