import torch
import triton
import triton.language as tl

# ──────────────────────────────────────────────────────────────
# FORWARD PASS KERNELS
# ──────────────────────────────────────────────────────────────


@triton.jit
def fwd_accumulate_kernel(
    a_ptr,
    b_ptr,
    chunk_a_ptr,
    chunk_c_ptr,
    T,
    D,
    stride_a_b,
    stride_a_t,
    stride_a_d,
    stride_b_b,
    stride_b_t,
    stride_b_d,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)

    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    # Chunk boundaries
    t_start = pid_t * BLOCK_SIZE_T

    # Initialize local carries
    a_acc = tl.full((BLOCK_SIZE_D,), 1.0, dtype=tl.float32)
    c_acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    a_base = a_ptr + pid_b * stride_a_b + d_offsets * stride_a_d
    b_base = b_ptr + pid_b * stride_b_b + d_offsets * stride_b_d

    for i in range(BLOCK_SIZE_T):
        t = t_start + i
        if t < T:
            a = tl.load(a_base + t * stride_a_t, mask=d_mask, other=1.0).to(tl.float32)
            b = tl.load(b_base + t * stride_b_t, mask=d_mask, other=0.0).to(tl.float32)

            # Combine: (a_acc, c_acc) o (a, b) = (a * a_acc, a * c_acc + b)
            c_acc = a * c_acc + b
            a_acc = a * a_acc

    # Store chunk summaries
    # chunk_ptr: [B, num_chunks, D]
    num_chunks = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    summary_idx = pid_b * (num_chunks * D) + pid_t * D + d_offsets
    tl.store(chunk_a_ptr + summary_idx, a_acc, mask=d_mask)
    tl.store(chunk_c_ptr + summary_idx, c_acc, mask=d_mask)


@triton.jit
def fwd_global_scan_kernel(
    chunk_a_ptr,
    chunk_c_ptr,
    chunk_carries_ptr,
    c_0_ptr,
    num_chunks,
    D,
    stride_c0_b,
    stride_c0_d,
    HAS_C_0: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    # Initial carry
    carry = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    if HAS_C_0:
        c0_ptrs = c_0_ptr + pid_b * stride_c0_b + d_offsets * stride_c0_d
        carry = tl.load(c0_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Base pointers for chunk summaries
    chunk_base = pid_b * (num_chunks * D) + d_offsets

    for j in range(num_chunks):
        # Store carry into chunk j (this is c_{j-1})
        tl.store(chunk_carries_ptr + chunk_base + j * D, carry, mask=d_mask)

        # Load chunk summary
        a_sum = tl.load(chunk_a_ptr + chunk_base + j * D, mask=d_mask, other=1.0).to(tl.float32)
        c_sum = tl.load(chunk_c_ptr + chunk_base + j * D, mask=d_mask, other=0.0).to(tl.float32)

        # Update carry for chunk j+1
        carry = a_sum * carry + c_sum


@triton.jit
def fwd_combine_kernel(
    a_ptr,
    b_ptr,
    chunk_carries_ptr,
    c_out_ptr,
    T,
    D,
    stride_a_b,
    stride_a_t,
    stride_a_d,
    stride_b_b,
    stride_b_t,
    stride_b_d,
    stride_c_b,
    stride_c_t,
    stride_c_d,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)

    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    num_chunks = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    t_start = pid_t * BLOCK_SIZE_T

    # Load initial carry for this chunk
    carry_idx = pid_b * (num_chunks * D) + pid_t * D + d_offsets
    carry = tl.load(chunk_carries_ptr + carry_idx, mask=d_mask, other=0.0).to(tl.float32)

    a_base = a_ptr + pid_b * stride_a_b + d_offsets * stride_a_d
    b_base = b_ptr + pid_b * stride_b_b + d_offsets * stride_b_d
    c_out_base = c_out_ptr + pid_b * stride_c_b + d_offsets * stride_c_d

    for i in range(BLOCK_SIZE_T):
        t = t_start + i
        if t < T:
            a = tl.load(a_base + t * stride_a_t, mask=d_mask, other=1.0).to(tl.float32)
            b = tl.load(b_base + t * stride_b_t, mask=d_mask, other=0.0).to(tl.float32)

            carry = a * carry + b
            tl.store(c_out_base + t * stride_c_t, carry, mask=d_mask)


# ──────────────────────────────────────────────────────────────
# BACKWARD PASS KERNELS
# ──────────────────────────────────────────────────────────────


@triton.jit
def bwd_accumulate_kernel(
    a_ptr,
    grad_c_out_ptr,
    chunk_a_prod_ptr,
    chunk_g_sum_ptr,
    T,
    D,
    stride_a_b,
    stride_a_t,
    stride_a_d,
    stride_g_b,
    stride_g_t,
    stride_g_d,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)

    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    t_start = pid_t * BLOCK_SIZE_T
    t_end = tl.minimum(t_start + BLOCK_SIZE_T, T)

    a_prod = tl.full((BLOCK_SIZE_D,), 1.0, dtype=tl.float32)
    g_sum = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    a_base = a_ptr + pid_b * stride_a_b + d_offsets * stride_a_d
    g_base = grad_c_out_ptr + pid_b * stride_g_b + d_offsets * stride_g_d

    # Reverse sequential accumulation for chunk summary
    # grad_c_start = (g_start + a_start+1*g_start+1 + ...) + (a_start+1*...*a_end) * grad_c_end
    # We iterate from t_end-1 down to t_start
    for i in range(t_end - t_start - 1, -1, -1):
        t = t_start + i
        g = tl.load(g_base + t * stride_g_t, mask=d_mask, other=0.0).to(tl.float32)

        # Multiplier is a_{t+1}. If t is T-1, multiplier is 1.0 (or 0 if we assume grad_c_T=0)
        # Actually, for the very last token in sequence, grad_c_T is 0.
        a_next = tl.full((BLOCK_SIZE_D,), 1.0, dtype=tl.float32)
        if t + 1 < T:
            a_next = tl.load(a_base + (t + 1) * stride_a_t, mask=d_mask, other=1.0).to(tl.float32)

        # combine: g_sum = g + a_next * g_sum, a_prod = a_next * a_prod
        g_sum = g + a_next * g_sum
        a_prod = a_next * a_prod

    num_chunks = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    summary_idx = pid_b * (num_chunks * D) + pid_t * D + d_offsets
    tl.store(chunk_a_prod_ptr + summary_idx, a_prod, mask=d_mask)
    tl.store(chunk_g_sum_ptr + summary_idx, g_sum, mask=d_mask)


@triton.jit
def bwd_global_scan_kernel(
    chunk_a_prod_ptr,
    chunk_g_sum_ptr,
    chunk_grad_carries_ptr,
    num_chunks,
    D,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    grad_carry = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    chunk_base = pid_b * (num_chunks * D) + d_offsets

    # Scan from last chunk to first
    for j in range(num_chunks - 1, -1, -1):
        # Store carry into chunk j (this is grad_c_{chunk_j_end})
        tl.store(chunk_grad_carries_ptr + chunk_base + j * D, grad_carry, mask=d_mask)

        a_prod = tl.load(chunk_a_prod_ptr + chunk_base + j * D, mask=d_mask, other=1.0).to(
            tl.float32
        )
        g_sum = tl.load(chunk_g_sum_ptr + chunk_base + j * D, mask=d_mask, other=0.0).to(tl.float32)

        # Update carry for chunk j-1
        # grad_c_{t_start_of_chunk_j} = g_sum_chunk_j + a_prod_chunk_j * grad_c_{t_end_of_chunk_j}
        grad_carry = g_sum + a_prod * grad_carry


@triton.jit
def bwd_combine_kernel(
    a_ptr,
    c_out_ptr,
    c_0_ptr,
    grad_c_out_ptr,
    chunk_grad_carries_ptr,
    grad_a_ptr,
    grad_b_ptr,
    grad_c_0_ptr,
    T,
    D,
    stride_a_b,
    stride_a_t,
    stride_a_d,
    stride_c_b,
    stride_c_t,
    stride_c_d,
    stride_g_b,
    stride_g_t,
    stride_g_d,
    stride_gb_b,
    stride_gb_t,
    stride_gb_d,
    stride_c0_b,
    stride_c0_d,
    HAS_C_0: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)

    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    num_chunks = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    t_start = pid_t * BLOCK_SIZE_T
    t_end = tl.minimum(t_start + BLOCK_SIZE_T, T)

    # Load initial gradient carry (this is grad_c_{t_end})
    # This was computed as grad_c_end in Pass 2.
    grad_at_tend = tl.load(
        chunk_grad_carries_ptr + pid_b * (num_chunks * D) + pid_t * D + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    a_base = a_ptr + pid_b * stride_a_b + d_offsets * stride_a_d
    c_out_base = c_out_ptr + pid_b * stride_c_b + d_offsets * stride_c_d
    g_base = grad_c_out_ptr + pid_b * stride_g_b + d_offsets * stride_g_d
    ga_base = grad_a_ptr + pid_b * stride_a_b + d_offsets * stride_a_d
    gb_base = grad_b_ptr + pid_b * stride_gb_b + d_offsets * stride_gb_d

    # running_grad enters index t as a_{t+1} * grad_c_{t+1}
    # For the very last token in chunk t=t_end-1, we need a_{t_end} * grad_c_{t_end}
    a_tend = tl.full((BLOCK_SIZE_D,), 1.0, dtype=tl.float32)
    if t_end < T:
        a_tend = tl.load(a_base + t_end * stride_a_t, mask=d_mask, other=1.0).to(tl.float32)

    running_grad = a_tend * grad_at_tend

    # Reverse scan within chunk
    for i in range(t_end - t_start - 1, -1, -1):
        t = t_start + i
        g_out_t = tl.load(g_base + t * stride_g_t, mask=d_mask, other=0.0).to(tl.float32)

        # grad_c_t = g_out_t + a_{t+1} * grad_c_{t+1}
        # In our loop, running_grad is always (a_{t+1} * grad_c_{t+1})
        grad_c_t = g_out_t + running_grad

        # Store results
        # grad_b_t = grad_c_t
        tl.store(gb_base + t * stride_gb_t, grad_c_t, mask=d_mask)

        # grad_a_t = c_{t-1} * grad_c_t
        c_prev = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
        if t > 0:
            c_prev = tl.load(c_out_base + (t - 1) * stride_c_t, mask=d_mask, other=0.0).to(
                tl.float32
            )
        elif HAS_C_0:
            c_prev = tl.load(
                c_0_ptr + pid_b * stride_c0_b + d_offsets * stride_c0_d, mask=d_mask, other=0.0
            ).to(tl.float32)

        tl.store(ga_base + t * stride_a_t, c_prev * grad_c_t, mask=d_mask)

        # update running_grad for the next iteration (t-1)
        # new running_grad = a_t * grad_c_t
        a_t = tl.load(a_base + t * stride_a_t, mask=d_mask, other=1.0).to(tl.float32)
        running_grad = a_t * grad_c_t

    # Final carry for d_c0 if pid_t == 0
    if pid_t == 0 and HAS_C_0:
        # After loop for t=0, running_grad is a_0 * grad_c_0
        tl.store(
            grad_c_0_ptr + pid_b * stride_c0_b + d_offsets * stride_c0_d, running_grad, mask=d_mask
        )


# ──────────────────────────────────────────────────────────────
# PYTORCH WRAPPER
# ──────────────────────────────────────────────────────────────


class DSRNScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c_0=None):
        B, T, D = a.shape
        device = a.device

        a = a.contiguous()
        b = b.contiguous()
        if c_0 is not None:
            c_0 = c_0.contiguous()

        c_out = torch.empty_like(a)

        BLOCK_SIZE_T = 64
        BLOCK_SIZE_D = triton.next_power_of_2(min(128, D))
        num_chunks = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T

        # Temporary workspace
        chunk_a = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)
        chunk_c = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)
        chunk_carries = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)

        # Pass 1: Accumulate
        grid1 = (B, triton.cdiv(D, BLOCK_SIZE_D), num_chunks)
        fwd_accumulate_kernel[grid1](
            a,
            b,
            chunk_a,
            chunk_c,
            T,
            D,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            BLOCK_SIZE_D,
            BLOCK_SIZE_T,
        )

        # Pass 2: Global Scan
        grid2 = (B, triton.cdiv(D, BLOCK_SIZE_D))
        fwd_global_scan_kernel[grid2](
            chunk_a,
            chunk_c,
            chunk_carries,
            c_0,
            num_chunks,
            D,
            c_0.stride(0) if c_0 is not None else 0,
            c_0.stride(1) if c_0 is not None else 0,
            HAS_C_0=(c_0 is not None),
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Pass 3: Combine
        fwd_combine_kernel[grid1](
            a,
            b,
            chunk_carries,
            c_out,
            T,
            D,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c_out.stride(0),
            c_out.stride(1),
            c_out.stride(2),
            BLOCK_SIZE_D,
            BLOCK_SIZE_T,
        )

        ctx.save_for_backward(a, c_out, c_0)
        ctx.BLOCK_SIZE_T = BLOCK_SIZE_T
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D

        return c_out

    @staticmethod
    def backward(ctx, grad_c_out):
        a, c_out, c_0 = ctx.saved_tensors
        B, T, D = a.shape
        device = a.device

        grad_c_out = grad_c_out.contiguous()
        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(a)
        grad_c_0 = torch.zeros_like(c_0) if c_0 is not None else None

        BLOCK_SIZE_T = ctx.BLOCK_SIZE_T
        BLOCK_SIZE_D = ctx.BLOCK_SIZE_D
        num_chunks = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T

        chunk_grad_a = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)
        chunk_grad_x = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)
        chunk_grad_carries = torch.empty((B, num_chunks, D), device=device, dtype=torch.float32)

        grid1 = (B, triton.cdiv(D, BLOCK_SIZE_D), num_chunks)

        # Pass 1: Accumulate
        bwd_accumulate_kernel[grid1](
            a,
            grad_c_out,
            chunk_grad_a,
            chunk_grad_x,
            T,
            D,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            grad_c_out.stride(0),
            grad_c_out.stride(1),
            grad_c_out.stride(2),
            BLOCK_SIZE_D,
            BLOCK_SIZE_T,
        )

        # Pass 2: Global Scan
        grid2 = (B, triton.cdiv(D, BLOCK_SIZE_D))
        bwd_global_scan_kernel[grid2](
            chunk_grad_a, chunk_grad_x, chunk_grad_carries, num_chunks, D, BLOCK_SIZE_D
        )

        # Pass 3: Combine
        bwd_combine_kernel[grid1](
            a,
            c_out,
            c_0,
            grad_c_out,
            chunk_grad_carries,
            grad_a,
            grad_b,
            grad_c_0,
            T,
            D,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            c_out.stride(0),
            c_out.stride(1),
            c_out.stride(2),
            grad_c_out.stride(0),
            grad_c_out.stride(1),
            grad_c_out.stride(2),
            grad_b.stride(0),
            grad_b.stride(1),
            grad_b.stride(2),
            c_0.stride(0) if c_0 is not None else 0,
            c_0.stride(1) if c_0 is not None else 0,
            HAS_C_0=(c_0 is not None),
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_T=BLOCK_SIZE_T,
        )

        return grad_a, grad_b, grad_c_0


def triton_dsrn_parallel_scan(g_t, m_t, c_0=None):
    orig_dtype = g_t.dtype
    a = (1.0 - g_t).float()
    b = (g_t * m_t).float()
    if c_0 is not None:
        c_0 = c_0.float()

    out = DSRNScanTriton.apply(a, b, c_0)
    return out.to(orig_dtype)
