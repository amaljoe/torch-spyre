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

Usage: python check.py <name> [kernel_dir]    (default: build/<name>)

Run it in a fresh process per example: a device fault leaves the stream in an
error state for the rest of the process.
"""

import sys
from pathlib import Path

import spyre_cli
import torch
from cases import EXAMPLES, RTOL
from references import DTYPES, expected, make_inputs


def main():
    name = sys.argv[1]
    ex = EXAMPLES[name]
    if len(sys.argv) > 2:
        path = Path(sys.argv[2])
    else:
        path = Path(__file__).parent / "build" / name

    inputs = make_inputs(name)
    out = torch.empty(ex.output, dtype=DTYPES[ex.dtype], device="spyre")

    runner = spyre_cli.launch(*[t.to("spyre") for t in inputs], out, path=path)
    got = out.cpu()
    del runner

    want = expected(name, inputs)
    max_diff = (got.float() - want.float()).abs().max().item()
    ok = torch.allclose(got, want, atol=ex.atol, rtol=RTOL)
    print(f"{name}: {'PASS' if ok else 'FAIL'} max|diff|={max_diff}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
