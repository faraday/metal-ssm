# Metal SSM

Metal SSM is a small, inference-only PyTorch reference implementation of the Mamba selective-scan recurrence for Apple Silicon. It uses a native Metal kernel with a PyTorch MPS bridge. This was used with PyTorch on an Apple M1 Max.

This is an experimental reference implementation with a restricted tensor contract. It is not a general drop-in replacement for Mamba's selective scan and does not implement autograd or training.

## Requirements

- Apple Silicon Mac with a working PyTorch MPS backend
- Python 3.10 or newer
- PyTorch 2.0 or newer with MPS support
- Xcode Command Line Tools with a C++20-capable Clang and the macOS Metal SDK

Install PyTorch and the build tools in the environment you intend to use. Then build the extension without pip's isolated build environment, so it uses the same PyTorch headers as the runtime:

```bash
git clone https://github.com/faraday/metal_ssm.git
cd metal_ssm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch setuptools wheel ninja
python -m pip install --no-build-isolation -e .
```

## Example

The wrapper accepts `u` and `delta` in `[B, D, L]` layout, `A` in `[D, N]`, and variable `B` and `C` in `[B, N, L]`. The example uses float32 MPS tensors and enables the delta softplus transform:

```python
import torch
from metal_ssm import metal_selective_scan_fn

batch, dim, length, state_size = 1, 64, 1024, 16
device = "mps"

u = torch.randn(batch, dim, length, device=device)
delta = torch.randn(batch, dim, length, device=device)
A = -torch.rand(dim, state_size, device=device)
B = torch.randn(batch, state_size, length, device=device)
C = torch.randn(batch, state_size, length, device=device)

with torch.inference_mode():
    out = metal_selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        delta_softplus=True,
    )

assert out.shape == (batch, dim, length)
```

`A` is the already transformed state-transition matrix (typically negative); pass `A`, not raw `A_log`. The wrapper converts its tensor inputs to float32 for computation, then casts `out` back to `u.dtype`.

## Input and output contract

All provided tensors must already be on the MPS device; the wrapper does not move tensors between devices.

| Argument | Shape | Required | Notes |
| --- | --- | --- | --- |
| `u`, `delta` | `[B, D, L]` | Yes | MPS tensors; `delta` may be raw when `delta_softplus=True`. |
| `A` | `[D, N]` | Yes | Already transformed, usually negative. |
| `B`, `C` | `[B, N, L]` | Yes | Sequence-dependent tensors. |
| `D` | `[D]` | No | Skip-connection weights. |
| `z` | `[B, D, L]` | No | Enables SiLU gating when provided. |
| `delta_bias` | `[D]` | No | Added to `delta` before optional softplus. |

The default result is `out` with shape `[B, D, L]`. Set `return_last_state=True` to receive `(out, last_state)`, where `last_state` has shape `[B, D, N]` and is returned as float32.

The shader stores the state in a fixed `float x[32]` array, so **`N` must be at most 32**. The current implementation does not check this limit at runtime; passing a larger `N` can access memory out of bounds. Keep `N ≤ 32`.

This interface supports only the layouts above; it does not accept every constant, grouped, or complex-valued B/C form supported by other selective-scan implementations. The native operation has no backward implementation, so outputs are not differentiable through this kernel.

## Related work

[SpeechLens](https://github.com/faraday/SpeechLens) is a related Apple Silicon project. It contains a separate Swift/MLX implementation of the same selective-scan recurrence; it does not use this repository's Python/PyTorch kernel.

## License

Metal SSM is licensed under the [Apache License 2.0](LICENSE).
