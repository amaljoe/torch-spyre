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


"""Launch one example kernel via the spyre_cli SDK and compare against CPU.

Usage: python check.py <name>

Run it in a fresh process per example: a device fault leaves the stream in an
error state for the rest of the process.
"""

import sys
from pathlib import Path

import spyre_cli
import torch
from cases import EXAMPLES, RTOL

REFERENCES = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "relu": torch.relu,
    "exp": torch.exp,
    "softmax": lambda a: torch.softmax(a, dim=-1),
    "mm": torch.mm,
}


def main():
    name = sys.argv[1]
    in_shapes, out_shape, atol = EXAMPLES[name]
    path = Path(__file__).parent / name

    torch.manual_seed(0)
    inputs = [torch.randn(shape, dtype=torch.float16) for shape in in_shapes]
    out = torch.empty(out_shape, dtype=torch.float16, device="spyre")

    runner = spyre_cli.launch(*[t.to("spyre") for t in inputs], out, path=path)
    got = out.cpu()
    del runner

    expected = REFERENCES[name](*[t.float() for t in inputs]).half()
    max_diff = (got.float() - expected.float()).abs().max().item()
    ok = torch.allclose(got, expected, atol=atol, rtol=RTOL)
    print(f"{name}: {'PASS' if ok else 'FAIL'} max|diff|={max_diff}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
