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


"""Integration tests: run every precompiled example on a Spyre device.

Each launch runs in its own process, since a device fault poisons the stream
for the rest of the process it happens in. Skipped when no device is present.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from cases import EXAMPLES, cli_args

DEVICE_ERRORS = ("RAS::", "StreamInErrorState", "DtException")


def _spyre_available():
    probe = "import torch; print(torch.accelerator.current_accelerator())"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "spyre" in result.stdout


pytestmark = pytest.mark.skipif(not _spyre_available(), reason="needs a Spyre device")


def _run(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLES_DIR,
        check=False,
    )
    return result, result.stdout + result.stderr


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_matches_cpu(name):
    result, output = _run([sys.executable, "check.py", name])
    assert result.returncode == 0, output


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_cli_launch(name):
    spyre = shutil.which("spyre")
    if spyre is None:
        pytest.skip("spyre CLI not installed")
    result, output = _run([spyre, "launch", *cli_args(name), name])
    assert result.returncode == 0, output
    assert not any(err in output for err in DEVICE_ERRORS), output
