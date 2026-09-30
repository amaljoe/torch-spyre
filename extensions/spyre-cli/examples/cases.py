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


"""Registry of the precompiled example kernels under ``examples/``.

Each entry names a kernel directory, the tensors ``spyre launch`` must be
given (inputs, then outputs, in kernel argument order), and how the result is
checked against CPU. Kept free of torch imports so tests can be collected on
machines without a Spyre device.
"""

# name -> (input shapes, output shape, atol). All tensors are fp16.
EXAMPLES = {
    "add": ([(512, 1024), (512, 1024)], (512, 1024), 1e-2),
    "sub": ([(512, 1024), (512, 1024)], (512, 1024), 1e-2),
    "mul": ([(512, 1024), (512, 1024)], (512, 1024), 2e-2),
    "relu": ([(512, 1024)], (512, 1024), 1e-2),
    "exp": ([(512, 1024)], (512, 1024), 5e-2),
    "softmax": ([(512, 1024)], (512, 1024), 1e-2),
    "mm": ([(512, 1024), (1024, 256)], (512, 256), 1.0),
}

RTOL = 1e-2
DTYPE = "fp16"


def tensor_spec(shape):
    """Format a shape as a ``spyre launch`` tensor argument, e.g. 512x1024@fp16."""
    return "x".join(str(d) for d in shape) + "@" + DTYPE


def cli_args(name):
    """The ``-i``/``-o`` arguments for launching example ``name``."""
    in_shapes, out_shape, _ = EXAMPLES[name]
    args = []
    for shape in in_shapes:
        args += ["-i", tensor_spec(shape)]
    return args + ["-o", tensor_spec(out_shape)]
