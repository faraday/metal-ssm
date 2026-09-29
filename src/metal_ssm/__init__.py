"""
Metal-accelerated selective scan for Mamba SSM on Apple Silicon.

Drop-in replacement for selective_scan_fn / selective_scan_ref from mamba-ssm.
"""
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn.functional as F

try:
    import selective_scan_metal_cpp
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False

_SHADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'selective_scan.metal')

def metal_selective_scan_fn(
    u, delta, A, B, C, D=None, z=None,
    delta_bias=None, delta_softplus=False, return_last_state=False,
):
    if not _HAS_METAL:
        raise RuntimeError("Metal extension not installed.")

    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    # Convert to format optimized for coalesced Metal memory access
    # u and delta start as [B, D, L]. We transpose to [B, L, D]
    delta_t = delta.transpose(1, 2).contiguous()  # [B, L, D]
    u_t = u.transpose(1, 2).contiguous() # [B, L, D]
    
    A_f = A.float().contiguous() # [D, N]
    B_t = B.float().transpose(1, 2).contiguous() # [B, L, N]
    C_t = C.float().transpose(1, 2).contiguous() # [B, L, N]

    # Optional arrays — if provided, must be in [B, L, D] or [D] format
    D_tensor = D.float().contiguous() if D is not None else torch.empty(0, device=u.device)
    z_tensor = z.float().transpose(1, 2).contiguous() if z is not None else torch.empty(0, device=u.device)

    # Calling the C++ extension. Output will be [B, L, D]
    results = selective_scan_metal_cpp.selective_scan_metal_fwd(
        delta_t,
        A_f,
        B_t,
        C_t,
        u_t,
        D_tensor,
        z_tensor,
        return_last_state,
        _SHADER_PATH,
    )

    # Transpose output back to [B, D, L]
    out = results[0].transpose(1, 2)
    last_state = results[1] if return_last_state else None

    out = out.to(dtype=dtype_in)

    if not return_last_state:
        return out
    return out, last_state

__all__ = ['metal_selective_scan_fn']
