# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from __init__ import metal_selective_scan_fn
except ImportError:
    from metal_ssm import metal_selective_scan_fn

def test():
    print("Creating tensors on MPS...")
    B, L, D, N = 1, 1024, 64, 16

    u = torch.randn(B, D, L, device="mps")
    delta = torch.randn(B, D, L, device="mps")
    A = -torch.rand(D, N, device="mps")
    B_seq = torch.randn(B, N, L, device="mps")
    C = torch.randn(B, N, L, device="mps")
    D_param = torch.randn(D, device="mps")
    z = torch.randn(B, D, L, device="mps")

    print("Running optimized metal_selective_scan_fn...")
    try:
        out = metal_selective_scan_fn(u, delta, A, B_seq, C, D=D_param, z=z)
        print("Success! Output shape:", out.shape)
        assert out.shape == (B, D, L)
        print("All tests passed.")
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test()
