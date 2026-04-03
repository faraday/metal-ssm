/******************************************************************************
 * Fused Mamba Selective Scan — Metal Compute Kernel
 *
 * Modified for fully coalesced memory access:
 * The inputs are passed in Transposed formats to ensure memory coalescing
 * across threads.
 *
 * Inputs (all float32, fully contiguous):
 *   deltaA:  [B, L, N, D]  — exp(delta ⊗ A), the decay factors
 *   deltaB_u: [B, L, N, D] — delta * B * u, the input injection
 *   C:       [B, L, N]     — readout projection (variable C)
 *   u:       [B, L, D]     — original input (for D skip connection)
 *   D_param: [D]           — skip connection weights (or nullptr)
 *   z:       [B, L, D]     — gating tensor for SiLU (or nullptr)
 *   out:     [B, L, D]     — output buffer
 *   last_state: [B, D, N]  — final state buffer (or nullptr)
 *
 * Grid:  (B * D, 1, 1)  — one thread per (batch, dim) pair
 * Adjacent threads process consecutive 'd' elements.
 * Since deltaA and deltaB_u lay out 'D' in the innermost dimension,
 * adjacent threads read memory addresses separated by 1 float (4 bytes).
 * This results in PERFECT memory coalescing.
 *
 * Copyright 2026. MIT License.
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// Sigmoid helper
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// SiLU (Swish) helper
inline float silu(float x) {
    return x * sigmoid(x);
}

kernel void selective_scan_fwd(
    device const float* deltaA      [[buffer(0)]],   // [B, L, N, D]
    device const float* deltaB_u    [[buffer(1)]],   // [B, L, N, D]
    device const float* C           [[buffer(2)]],   // [B, L, N]
    device const float* u           [[buffer(3)]],   // [B, L, D]
    device const float* D_param     [[buffer(4)]],   // [D]
    device const float* z           [[buffer(5)]],   // [B, L, D]
    device float* out               [[buffer(6)]],   // [B, L, D]
    device float* last_state        [[buffer(7)]],   // [B, D, N]
    constant uint& B_size           [[buffer(8)]],
    constant uint& D_size           [[buffer(9)]],
    constant uint& L_size           [[buffer(10)]],
    constant uint& N_size           [[buffer(11)]],
    constant uint& has_D            [[buffer(12)]],
    constant uint& has_z            [[buffer(13)]],
    constant uint& save_last_state  [[buffer(14)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint total_bd = B_size * D_size;
    if (tid >= total_bd) return;

    // Adjacent threads map to adjacent 'd' values
    uint b = tid / D_size;
    uint d = tid % D_size;

    // State vector in registers — N up to 32
    float x[32];
    for (uint n = 0; n < N_size; n++) {
        x[n] = 0.0f;
    }

    // Base pointer offsets for the current batch 'b'
    // For [B, L, N, D]:
    uint base_blnd = b * (L_size * N_size * D_size);
    // For [B, L, N]:
    uint base_bln = b * (L_size * N_size);
    // For [B, L, D]:
    uint base_bld = b * (L_size * D_size);

    for (uint i = 0; i < L_size; i++) {
        // Offset for current step i
        uint offset_blnd_i = base_blnd + i * (N_size * D_size) + d;

        // State update: x[n] = deltaA * x[n] + deltaB_u
        for (uint n = 0; n < N_size; n++) {
            uint idx = offset_blnd_i + n * D_size;
            float dA = deltaA[idx];
            float dBu = deltaB_u[idx];
            x[n] = dA * x[n] + dBu;
        }

        // Readout y = sum_n(x[n] * C[b, l, n])
        float y = 0.0f;
        uint offset_bln_i = base_bln + i * N_size;
        for (uint n = 0; n < N_size; n++) {
            float c_val = C[offset_bln_i + n];
            y += x[n] * c_val;
        }

        // Apply D skip connection
        uint offset_bld_i = base_bld + i * D_size + d;
        if (has_D) {
            y += u[offset_bld_i] * D_param[d];
        }

        // Apply z gating
        if (has_z) {
            y *= silu(z[offset_bld_i]);
        }

        // Write output [B, L, D]
        out[offset_bld_i] = y;
    }

    // Save final state if needed
    if (save_last_state) {
        // last_state layout: [B, D, N]
        uint base_state = b * D_size * N_size + d * N_size;
        for (uint n = 0; n < N_size; n++) {
            last_state[base_state + n] = x[n];
        }
    }
}
