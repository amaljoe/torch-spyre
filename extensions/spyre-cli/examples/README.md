# spyre-cli examples

Precompiled kernel directories you can launch with `spyre launch` without
compiling anything first. Each directory is the unmodified output of
`torch.compile(fn, backend="inductor")` for one op: `bundle.mlir` and
`sdsc_*.json` are the SDSC bundle, `spyreCodeDir/` is what `spyre launch`
actually loads.

All tensors are fp16. Run from this directory on a machine with a Spyre card:

| Example | Op | Command |
|---|---|---|
| `add` | `a + b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 add` |
| `sub` | `a - b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 sub` |
| `mul` | `a * b` | `spyre launch -i 512x1024@fp16 -i 512x1024@fp16 -o 512x1024@fp16 mul` |
| `relu` | `torch.relu(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 relu` |
| `exp` | `torch.exp(a)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 exp` |
| `softmax` | `torch.softmax(a, dim=-1)` | `spyre launch -i 512x1024@fp16 -o 512x1024@fp16 softmax` |
| `mm` | `a @ b` | `spyre launch -i 512x1024@fp16 -i 1024x256@fp16 -o 512x256@fp16 mm` |

The CLI fills every input with ones, so e.g. `add` prints a tensor of 2s.

## Testing

`cases.py` is the registry of examples (shapes and tolerances). `check.py
<name>` launches one example through the SDK with random inputs and compares
the result against CPU. `tests/test_examples.py` runs both `check.py` and the
real `spyre launch` CLI for every example, each in a fresh process:

```bash
python3 -m pytest tests/test_examples.py -v
```

The tests skip themselves when no Spyre device is present.

## Caveats

- The kernels are tied to the toolchain that built them (torch-spyre
  `e2028e39`, image `icr.io/ai_sw_accel/2.0/torch-spyre:latest` as of
  2026-09-30). Regenerate them with `torch.compile` if the runtime stops
  loading them.
- spyre-cli allocates every tensor with the default device layout. A kernel
  compiled for any other layout gives wrong results with no error. For example
  `a.sum(dim=-1)` writes a `(512,)` output with `device_size=[1, 512, 64]`,
  where the default is `[8, 64]`, so it is not included here.
